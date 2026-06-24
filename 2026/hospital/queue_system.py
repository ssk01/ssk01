#!/usr/bin/env python3
"""
医院排队叫号系统 — 核心伪代码
================================
此文件是完整系统设计的伪代码实现，对应设计报告中所有核心模块。
语言风格：类 Python 伪代码，侧重逻辑清晰而非可编译。

运行演示: python3 main.py
"""

import uuid, time, threading, json, heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict

# =============================================================================
# 1. 常量定义
# =============================================================================

class Priority(Enum):
    EMERGENCY  = 0   # 急诊
    RETURN     = 1   # 回诊
    ELDERLY    = 2   # 老年
    NORMAL     = 3   # 正常预约
    LATE       = 4   # 迟到
    WALK_IN    = 5   # 现场挂号

class TicketStatus(Enum):
    WAITING  = "WAITING"
    CALLED   = "CALLED"
    SERVING  = "SERVING"
    DONE     = "DONE"
    NOSHOW   = "NOSHOW"

class DoctorStatus(Enum):
    OFFLINE = "OFFLINE"
    IDLE    = "IDLE"
    BUSY    = "BUSY"

CALL_TIMEOUT_SEC = 180   # 叫号超时 3 分钟
REJOIN_THRESHOLD = 600   # 过号 10 分钟排队尾

# =============================================================================
# 2. 数据模型
# =============================================================================

@dataclass
class Patient:
    patient_id: str
    name: str
    age: int
    is_emergency: bool = False

@dataclass
class Doctor:
    doctor_id: str
    name: str
    department_id: str
    room_no: str
    status: DoctorStatus = DoctorStatus.IDLE
    current_ticket_id: Optional[str] = None

@dataclass
class Department:
    department_id: str
    name: str
    prefix: str              # 叫号前缀 "A"
    current_number: int = 0  # 当日已发号数

@dataclass
class Ticket:
    ticket_id: str
    patient_id: str
    department_id: str
    queue_number: str
    priority: Priority
    check_in_time: float
    status: TicketStatus = TicketStatus.WAITING
    doctor_id: Optional[str] = None
    called_time: Optional[float] = None

# =============================================================================
# 3. 模拟存储层 (生产环境用 Redis + PostgreSQL)
# =============================================================================

class Storage:
    """模拟 Redis + PostgreSQL 的混合存储"""

    def __init__(self):
        self.tickets: Dict[str, Ticket] = {}
        self.doctors: Dict[str, Doctor] = {}
        self.departments: Dict[str, Department] = {}
        # 科室队列: {dept_id: [(score, ticket_id)]}  — 模拟 Redis Sorted Set
        self.queues: Dict[str, list] = {}
        # 分布式锁
        self.locks: Dict[str, bool] = {}
        # Pub/Sub 订阅者
        self.subscribers: Dict[str, list] = {}
        self.lock = threading.Lock()

    def acquire_lock(self, key: str) -> bool:
        with self.lock:
            if self.locks.get(key):
                return False
            self.locks[key] = True
            return True

    def release_lock(self, key: str):
        with self.lock:
            self.locks[key] = False

    def zadd(self, queue_key: str, score: int, ticket_id: str):
        if queue_key not in self.queues:
            self.queues[queue_key] = []
        heapq.heappush(self.queues[queue_key], (score, ticket_id))

    def zpop(self, queue_key: str) -> Optional[str]:
        """原子弹出最小 score"""
        if queue_key not in self.queues or not self.queues[queue_key]:
            return None
        score, ticket_id = heapq.heappop(self.queues[queue_key])
        return ticket_id

    def zcard(self, queue_key: str) -> int:
        return len(self.queues.get(queue_key, []))

    def zrange(self, queue_key: str, start: int, end: int) -> list:
        q = sorted(self.queues.get(queue_key, []))
        return [tid for _, tid in q[start:end+1]]

    def publish(self, channel: str, message: dict):
        if channel in self.subscribers:
            for callback in self.subscribers[channel]:
                callback(message)

    def subscribe(self, channel: str, callback):
        self.subscribers.setdefault(channel, []).append(callback)

    # ---------- 模拟 Lua 原子脚本 ----------
    def atomic_pop_next(self, queue_key: str) -> Optional[str]:
        """模拟 Redis Lua: 弹出 + 过号检查 + skip NOSHOW"""
        ticket_id = self.zpop(queue_key)
        if ticket_id and ticket_id in self.tickets:
            ticket = self.tickets[ticket_id]
            if ticket.status == TicketStatus.CALLED:
                elapsed = time.time() - (ticket.called_time or 0)
                if elapsed > CALL_TIMEOUT_SEC:
                    ticket.status = TicketStatus.NOSHOW
                    # 已过号，递归取下一个
                    return self.atomic_pop_next(queue_key)
        return ticket_id


# =============================================================================
# 4. 核心业务逻辑
# =============================================================================

SCORE_BASE = 10**15  # 优先级乘数必须远大于时间戳毫秒值(~10^12)

class HospitalQueueSystem:
    """
    医院排队叫号系统主类
    ====================
    对应设计报告中的三个核心服务:
      - CheckInService   (签到)
      - CallService      (叫号)
      - DisplayService   (大屏推送)
    """

    def __init__(self):
        self.store = Storage()

    # =====================================================================
    # 4.1 初始化数据
    # =====================================================================

    def init_department(self, dept_id: str, name: str, prefix: str):
        self.store.departments[dept_id] = Department(dept_id, name, prefix)

    def init_doctor(self, doc_id: str, name: str, dept_id: str, room: str):
        self.store.doctors[doc_id] = Doctor(doc_id, name, dept_id, room)

    # =====================================================================
    # 4.2 优先级计算
    # =====================================================================

    def calc_priority(self, patient: Patient, is_return: bool) -> Priority:
        if patient.is_emergency:
            return Priority.EMERGENCY
        if is_return:
            return Priority.RETURN
        if patient.age >= 80:
            return Priority.ELDERLY
        return Priority.NORMAL

    # =====================================================================
    # 4.3 签到取号
    # =====================================================================

    def check_in(self, patient: Patient, dept_id: str, is_return: bool = False) -> Ticket:
        dept = self.store.departments.get(dept_id)
        if not dept:
            raise ValueError(f"科室 {dept_id} 不存在")

        priority = self.calc_priority(patient, is_return)

        ticket_id = str(uuid.uuid4())[:8]

        # 号源计数器 (Redis INCR 模拟)
        lock_key = f"dept_counter:{dept_id}"
        while not self.store.acquire_lock(lock_key):
            time.sleep(0.01)

        dept.current_number += 1
        queue_number = f"{dept.prefix}{dept.current_number:03d}"
        self.store.release_lock(lock_key)

        # 计算 score
        now_ns = time.time_ns()
        score = priority.value * SCORE_BASE + now_ns

        ticket = Ticket(
            ticket_id=ticket_id,
            patient_id=patient.patient_id,
            department_id=dept_id,
            queue_number=queue_number,
            priority=priority,
            check_in_time=time.time(),
        )

        # 写入存储
        self.store.tickets[ticket_id] = ticket
        queue_key = f"queue:dept:{dept_id}"
        self.store.zadd(queue_key, score, ticket_id)

        # 广播队列更新
        self.store.publish(f"display:{dept_id}", {
            "type": "QUEUE_UPDATE",
            "total_waiting": self.store.zcard(queue_key),
        })

        return ticket

    # =====================================================================
    # 4.4 医生叫下一位 (核心)
    # =====================================================================

    def call_next(self, doctor_id: str) -> dict:
        doctor = self.store.doctors.get(doctor_id)
        if not doctor:
            raise ValueError(f"医生 {doctor_id} 不存在")

        # 分布式锁：同一科室同时只能一个叫号
        lock_key = f"call:{doctor.department_id}"
        while not self.store.acquire_lock(lock_key):
            time.sleep(0.01)

        try:
            return self._call_next_internal(doctor)
        finally:
            self.store.release_lock(lock_key)

    def _call_next_internal(self, doctor: Doctor) -> dict:
        if doctor.status == DoctorStatus.BUSY and doctor.current_ticket_id:
            raise RuntimeError("请先完成当前就诊")

        queue_key = f"queue:dept:{doctor.department_id}"

        # 原子出队
        ticket_id = self.store.atomic_pop_next(queue_key)
        if ticket_id is None:
            doctor.status = DoctorStatus.IDLE
            doctor.current_ticket_id = None
            return {"message": "当前无人等待"}

        ticket = self.store.tickets[ticket_id]

        # 更新状态
        ticket.status = TicketStatus.CALLED
        ticket.doctor_id = doctor.doctor_id
        ticket.called_time = time.time()

        doctor.status = DoctorStatus.BUSY
        doctor.current_ticket_id = ticket_id

        # 构建大屏消息
        display_msg = {
            "type": "CALL",
            "ticket_id": ticket.ticket_id,
            "queue_number": ticket.queue_number,
            "room_no": doctor.room_no,
            "patient_name": "患者" + ticket.patient_id[:4],
            "timestamp": time.time(),
        }

        # 推送大屏
        self.store.publish(f"display:{doctor.department_id}", display_msg)

        # 异步 TTS (模拟)
        self._async_tts(display_msg)

        return {
            "ticket_id": ticket.ticket_id,
            "queue_number": ticket.queue_number,
            "patient_name": display_msg["patient_name"],
            "room_no": doctor.room_no,
        }

    # =====================================================================
    # 4.5 完成就诊 / 过号
    # =====================================================================

    def complete_consultation(self, doctor_id: str):
        doctor = self.store.doctors[doctor_id]
        if doctor.current_ticket_id:
            ticket = self.store.tickets[doctor.current_ticket_id]
            ticket.status = TicketStatus.DONE
        doctor.status = DoctorStatus.IDLE
        doctor.current_ticket_id = None

    def mark_no_show(self, doctor_id: str):
        doctor = self.store.doctors[doctor_id]
        if doctor.current_ticket_id:
            ticket = self.store.tickets[doctor.current_ticket_id]
            ticket.status = TicketStatus.NOSHOW
        doctor.status = DoctorStatus.IDLE
        doctor.current_ticket_id = None

    # =====================================================================
    # 4.6 过号重排
    # =====================================================================

    def rejoin_queue(self, ticket_id: str) -> Ticket:
        ticket = self.store.tickets.get(ticket_id)
        if not ticket or ticket.status != TicketStatus.CALLED:
            raise ValueError("仅已叫号未就诊可重排")

        elapsed = time.time() - (ticket.called_time or 0)

        # 3 分钟内不允许重排 (去趟厕所就回来了)
        if elapsed < CALL_TIMEOUT_SEC:
            raise ValueError(f"叫号不足3分钟({int(elapsed)}s)，请稍后再来")

        # 超 10 分钟排更靠后
        if elapsed > REJOIN_THRESHOLD:
            new_priority = min(ticket.priority.value + 2, 5)
        else:
            new_priority = ticket.priority.value

        score = new_priority * SCORE_BASE + time.time_ns()
        queue_key = f"queue:dept:{ticket.department_id}"

        ticket.status = TicketStatus.WAITING
        ticket.priority = Priority(new_priority)
        ticket.check_in_time = time.time()
        self.store.zadd(queue_key, score, ticket_id)

        self.store.publish(f"display:{ticket.department_id}", {
            "type": "QUEUE_UPDATE",
            "total_waiting": self.store.zcard(queue_key),
        })

        return ticket

    # =====================================================================
    # 4.7 大屏服务
    # =====================================================================

    def get_display_snapshot(self, dept_id: str) -> dict:
        """大屏初始化 / HTTP 降级轮询接口"""
        queue_key = f"queue:dept:{dept_id}"
        ticket_ids = self.store.zrange(queue_key, 0, 19)

        waiting_list = []
        for tid in ticket_ids:
            t = self.store.tickets.get(tid)
            if t:
                waiting_list.append({
                    "queue_number": t.queue_number,
                    "patient_id": t.patient_id,
                    "priority": t.priority.name,
                    "wait_minutes": int((time.time() - t.check_in_time) / 60),
                })

        # 查找当前正在叫号的
        current_call = None
        for t in self.store.tickets.values():
            if t.department_id == dept_id and t.status == TicketStatus.CALLED:
                doc = self.store.doctors.get(t.doctor_id or "")
                current_call = {
                    "queue_number": t.queue_number,
                    "room_no": doc.room_no if doc else "—",
                    "called_seconds_ago": int(time.time() - (t.called_time or 0)),
                }
                break

        return {
            "current_call": current_call,
            "waiting_list": waiting_list,
            "total_waiting": self.store.zcard(queue_key),
        }

    def subscribe_display(self, dept_id: str, callback):
        """WebSocket 订阅：大屏实时接收推送"""
        self.store.subscribe(f"display:{dept_id}", callback)

    # =====================================================================
    # 4.8 异步语音播报
    # =====================================================================

    def _async_tts(self, msg: dict):
        """模拟异步 TTS，实际生产中投递到消息队列由独立 TTS 服务处理"""
        threading.Thread(
            target=lambda: print(
                f"  🔊 TTS 播报: 请 {msg['queue_number']} 号到 {msg['room_no']} 诊室就诊"
            ),
            daemon=True,
        ).start()

    # =====================================================================
    # 4.9 预估等待时间
    # =====================================================================

    def estimate_wait(self, dept_id: str, priority: Priority) -> int:
        queue_key = f"queue:dept:{dept_id}"
        total = self.store.zcard(queue_key)
        avg_consult_min = 8  # 平均就诊 8 分钟
        return (total * avg_consult_min) // len(self._get_active_doctors(dept_id) or 1)

    def _get_active_doctors(self, dept_id: str) -> list:
        return [d for d in self.store.doctors.values()
                if d.department_id == dept_id and d.status != DoctorStatus.OFFLINE]


# =============================================================================
# 5. 演示程序
# =============================================================================

def demo():
    print("=" * 60)
    print("    🏥 医院排队叫号系统 — 演示")
    print("=" * 60)

    sys = HospitalQueueSystem()

    # --- 初始化科室 & 医生 ---
    sys.init_department("D001", "心血管内科", "A")
    sys.init_department("D002", "骨科", "B")

    sys.init_doctor("DOC01", "张医生", "D001", "301")
    sys.init_doctor("DOC02", "李医生", "D001", "302")
    sys.init_doctor("DOC03", "王医生", "D002", "401")

    # --- 大屏订阅 (模拟 WebSocket) ---
    def on_display_update(msg):
        if msg["type"] == "CALL":
            print(f"\n  📺 大屏更新 → 请 {msg['queue_number']} 到 {msg['room_no']}")
        elif msg["type"] == "QUEUE_UPDATE":
            print(f"  📊 队列更新 → 当前等待: {msg['total_waiting']} 人")

    sys.subscribe_display("D001", on_display_update)

    # --- 患者签到 ---
    print("\n--- 1. 患者签到 ---")
    patients = [
        Patient("P001", "张三", 35),
        Patient("P002", "李四", 82),
        Patient("P003", "王五", 45, is_emergency=True),
        Patient("P004", "赵六", 28),
    ]

    for p in patients:
        t = sys.check_in(p, "D001")
        print(f"  {p.name} → {t.queue_number} (优先级: {t.priority.name})")

    # --- 查看大屏 ---
    print("\n--- 2. 大屏快照 ---")
    snap = sys.get_display_snapshot("D001")
    print(f"  等待人数: {snap['total_waiting']}")
    for w in snap["waiting_list"]:
        print(f"    {w['queue_number']} | {w['priority']} | 等待 {w['wait_minutes']}min")

    # --- 医生叫号 ---
    print("\n--- 3. 张医生叫号 (应叫急诊王五) ---")
    r = sys.call_next("DOC01")
    print(f"  叫到: {r}")

    print("\n--- 4. 张医生完成王五就诊 ---")
    sys.complete_consultation("DOC01")

    print("\n--- 5. 张医生叫号 (应叫老年李四) ---")
    r = sys.call_next("DOC01")
    print(f"  叫到: {r}")

    print("\n--- 6. 李医生叫号 (应叫正常张三, 赵六排后面) ---")
    r = sys.call_next("DOC02")
    print(f"  叫到: {r}")

    # --- 过号重排 ---
    print("\n--- 7. 模拟过号: 李四被叫后3分钟未到 → 重新排队 ---")
    ticket_lisi = None
    for t in sys.store.tickets.values():
        if t.patient_id == "P002":
            ticket_lisi = t
            ticket_lisi.called_time = time.time() - 200  # 模拟已过 200s
            break

    sys.rejoin_queue(ticket_lisi.ticket_id)
    print(f"  李四重新入队 → 等待队列: {sys.store.zcard('queue:dept:D001')} 人 - 当前大屏快照:")
    snap = sys.get_display_snapshot("D001")
    for w in snap["waiting_list"]:
        print(f"    {w['queue_number']} | {w['priority']} | 等待 {w['wait_minutes']}min")

    print("\n" + "=" * 60)
    print("    ✅ 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo()
