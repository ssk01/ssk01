# Socrates

### Q: `Uniqlo& operator=(const Uniqlo& t) = delete;` 和原来的 `Uniqlo operator=(Uniqlo& t) = delete;` 差别在哪？需要再加 `(Uniqlo& t)` 或 `(Uniqlo t)` 重载吗？
- 差别有三点：① const 限定——`const Uniqlo&` 能同时匹配非 const 左值 / const 左值 / 右值，`Uniqlo&` 只能匹配非 const 左值；② 返回类型——规范 copy assignment 返回 `Uniqlo&`（支持链式 `a=b=c`），原版返回 `Uniqlo`（按值）；③ 错误提示——`const Uniqlo&` 删除后所有赋值路径都报清晰的 "use of deleted function"，`Uniqlo&` 对 const 对象会报 "no matching function"（语义混乱）。
- 不需要加重载。单个 `const Uniqlo&` 已覆盖所有左值/右值匹配路径；后续加 move 赋值 `operator=(Uniqlo&&)` 后，右值会优先走 move，const 右值才会落到被删的 copy 上，行为正确。
(2026-08-31)

### Q: move 构造用 `std::swap(u.t_, t_)` 会出什么问题？
- 模板类的成员函数是惰性实例化的：只要没人调用，编译错误/UB 不会暴露——旧 demo 全过，但新特性测试一触发就现形。
- `std::swap(u.t_, t_)` 在 move 构造里是错的：`t_` 此时尚未初始化，swap 会把未初始化的垃圾指针写进被移动对象 `u`，其析构 `delete` 垃圾指针 = UB（实测 SIGABRT）。正确写法：`t_(u.t_) { u.t_ = nullptr; }`——把指针搬走、源置空。
- move 赋值用 swap 恰好"能工作"（被移动方持有对方旧资源、由其析构收尾），但语义和 std 不一致（std 要求 moved-from 为空）；仍建议按标准语义写。
- macOS + 新版 Xcode 的 ASan 启动即挂是工具链已知问题，UBSan 正常可用。
(2026-08-31)

### Q: 值类别：为什么"命名变量"一定是左值？std::move/forward 到底是什么？
- 值类别是**表达式**的属性，不是对象的属性。对象有名字 → 用它名字写出的表达式永远是左值（有身份、有地址、持久存在）；临时量没名字 → 右值。
- `int&& r = 5;` 里 `&&` 是**类型**（右值引用类型）不是值类别；`r` 一旦绑定就是命名对象 = 左值。这就是 `std::move` 存在的意义——把命名左值转回右值表达式。
- 按值收参：实参左值 → 拷贝构造；实参右值 → 移动构造（`T(const T&)` vs `T(T&&)` 重载决议）。`int` 拷贝==移动所以无差别；unique_ptr 拷贝被删所以必须右值。
- `std::forward<T>(x)` = `static_cast<T&&>(x)`——一个 cast，输出类型**只由 T 决定**，与实参值类别无关。推导语境（T 从实参推）表现成"左保左/右保右"；显式指定语境（T 写死）普通类型就是 move。
- move 构造 `forward<Deleter>(u.get_deleter())`：get_deleter 返回左值引用，但 T 写死为普通类型 → cast 成右值 → 移动。**不能用 `std::move`**：无脑转右值，引用 deleter `D&` 会编译错；`forward<D&>` 折叠回 `D&` 保左。
(2026-08-31)

### Q: EBO 到底"零"了什么？
- "zero" 不是整个对象变 0 字节——是**空 deleter 贡献 0 字节**。`t_` 指针 8 字节是物理下限。
- 空类独立存在要 1 字节（两对象不能同址）；**空基类子对象可 0 字节**，与首个数据成员重叠（`&基类 == &t_`，实测同地址）。
- 继承式 `class Uniqlo : private Deleter { T* t_; }`：空 deleter 贴到 t_ 上 → sizeof == 8 == 裸指针。
- 传 deleter 给基类 = 构造初始化列表 `: Deleter(d)`；用 deleter = `static_cast<Deleter&>(*this)` 上转型。
- 现代 libc++ 不用继承，用 C++20 `[[no_unique_address]]` 成员属性，效果相同且支持引用 deleter（引用不能当基类）。
(2026-08-31)

### Q: swap 为什么必须连带交换 deleter？
- pointer 和 deleter 是一对（指针 → 释放方式）。只换指针不换 deleter，函数指针 deleter（`void(*)(T*)`）场景会错配：`free` 释放 `new` 内存 = UB（编译能过，类型系统抓不到，纯运行时炸）。
- 空 deleter 之间 swap = no-op（三次 move 全空），无害且免费——所以无条件写 `std::swap(ptr) + std::swap(deleter)` 两端通吃。
- `delete t_` 不会置空指针（悬空地址还在）；`delete nullptr` 是安全 no-op 不用判空。
(2026-08-31)
### Q: shy_ptr.h 能跑通 cppreference shared_ptr 页面那两个 demo 吗？
- 跑不通，且最致命的问题不是缺 API，而是**控制块存活条件判错**：`ShyPtrBase::expired()` 返回 `cb_cnt==0`（shy_ptr.h:28），而控制块 `base_` 的正确销毁条件是 **强引用与弱引用都为 0**（`count_==0 && cb_cnt==0`）。正常共享场景没有 weak（cb_cnt 恒 0），导致**任何一个 ShyPtr 析构都会删掉共享的控制块**，别的副本还在用 → 两个副本共享即崩溃（UBSan 实测 SIGTRAP，b 先析构删 base_、a 再摸已释放内存）。WeakPtr 析构同理反向错：弱引用清零但强引用活着时也会提前删 base_。
- 模板成员函数惰性实例化掩盖了一批**跨类访问 private** 的编译错：`WeakPtr::operator=(const ShyPtr<T>&)` 摸 `ShyPtr::base_`（无 friend，shy_ptr.h:127）、`ShyPtr(const WeakPtr<T>&)` 摸 `WeakPtr::base_`——只有用到对应成员才炸，和 Uniqlo 那次经验一模一样。
- 其它 API 层挡路：`get()`/`use_count()` 非 const（demo 的 print 收 const& 就编译错）；`lock()` 返回 `ShyPtr<T>&` 绑临时对象（悬垂）；无 `operator*`/`operator->`/`operator bool`；copy/move assign 不放旧引用（泄漏）且漏 `return *this`；对象删除 `count_--; if==0` 分两步，多线程可能双重 delete，应 `fetch_sub(1)==1`。
- 修好上述后 demo 还有差距：缺 make_shared；控制块拿 `Base*` 直接 delete（非虚析构下不会调 `~Derived`，输出与 std 不同）；enable_shared_from_this + alias 构造那个 demo 更远。
(2026-09-04 16:56)
### Q: 为什么 std::shared_ptr 能"记得"真实类型并正确析构，而 `Base* p=new Derived; delete p;` 不行？非虚析构下 delete 基类指针本来就不该调 ~Derived 吧？
- 对，第二点判断正确：`delete` 选析构**只认静态类型 + 是否虚**。`~Base` 虚 → vtable 分派到 ~Derived；非虚 → 编译期定死调 ~Base，不递归。所以 probe 只打 `Base::~Base` 是标准行为，不是 bug。
- shared_ptr "记得住"的关键是**记忆发生在构造期而非 delete 期**：`shared_ptr<Base> sp(new Derived)` 走模板构造 `template<class Y> shared_ptr(Y*)`，Y=Derived 在构造点是已知的；控制块存的是**类型擦除的 deleter**（把 p 当 Derived 保管），计数归零时按 Derived 删。`delete (Derived*)` 的静态类型本来就是 Derived，不需要 vtable，~Derived 跑完析构链自动调 ~Base。这就是 cppreference 敢写 "non-virtual destructor is OK here" 的原因。
- 普通写法在 new 的瞬间类型就坍缩成 Base* 了，delete 期信息不存在；shared_ptr 则是在信息还没丢的构造点把它抓进控制块。
- ShyPtr 丢信息的精确位置：`ShyPtr(T* ptr): base_(new ShyPtrBase<T>(ptr))` 里 T=Base，`new Derived` 进构造瞬间被隐式转成 Base*，入口即丢；`delete (T*)ptr_` 只能按 Base 删。修法二选一：① demo 里 ~Base 改虚（现有 delete Base* 也能靠 vtable 分派）；② 照 std 做类型擦除 deleter（demo 保持非虚也对）。虚析构与类型擦除是达成同一目标的两种独立手段。
- 追问"~Base 非虚时，~Derived 正常执行后会自动调 ~Base 吗"：**会，且与虚无关**。析构链是编译器保证的——~Derived 函数体跑完自动析构基类子对象调 ~Base。虚只决定"从基类指针 delete 时能不能调到 ~Derived"这一环；只要 ~Derived 被调到，~Base 必然跟着跑（虚析构场景里 ~Base 也正是靠这条链被自动调的）。所以类型擦除后按 Derived delete，输出会是 ~Derived 再 ~Base。
- 追问"模板把编译期类型绑进 delete 的 lambda，为什么叫类型擦除"：擦除的是**存储结构里的 Y**，不是删除代码。判定标准是"控制块类型里还找得到 Y 吗"——现在 `ShyPtrBase<T>` 是单一固定类型，成员只是 `void(*)(T*)` 函数指针，Y 只出现在构造点（按 Y 实例化出不同 lambda 塞进指针值），运行期 Y 从数据中消失、只剩不透明指针 → 类型转成行为再丢弃，即擦除。**为什么必须擦**：若控制块也带 Y（如 ShyPtrBase<T,Y>），`ShyPtr<Base>(new Derived)` 与 `new Base` 会得到不同类型控制块，而多个 `ShyPtr<Base>` 拷贝要共享同一个控制块对象——只有 Y 不进类型、T 固定，才能让同 T 不同 Y 的对象落到同一种控制块上。擦掉的是 Y（构造实参类型），T 仍在（ptr_/get 都是 T*），std 同样保留 T 只擦 deleter 类型。同一思想：std::function、shared_ptr 的 deleter/allocator、std::thread、std::any、虚函数 vtable。
(2026-09-04 17:28)
### Q: std::thread 的 callable 不是具体类型吗？跟类型擦除什么关系？为什么 std::thread 能把任意 callable+任意参数都存下来？
- `std::thread` 类本身**不是模板**（`class thread`），lambda/函数指针/仿函数构造出来都是同一类型——说明 callable 具体类型在存储层被擦掉了。擦除点在其模板构造函数 `template<class F, class... Args> thread(F&&, Args&&...)`：F 在入口已知并 decay（去引用/顶层cv、函数转函数指针、数组转指针，故传引用需显式 std::ref），随后把 F+args 装箱进堆上 `thread_data_impl<F,Args...>`，经基类虚函数 run() 调用。
- 为什么必须擦：底层线程入口签名固定（如 `void*(*)(void*)`），任意 callable+参数无法直接塞进去，只能装箱成统一形态+统一入口交给 OS。这和 deleter 的擦除是同一模式——类型在构造边界消费成统一运行时形态（虚函数或函数指针），存储层不再知道 F。虚函数是 C++ 内置擦除手段，deleter 用的函数指针是另一种形态；std::function 同理。
### Q: `const ShyPtr<T>`（todo：常量引用问题）该怎么解？拷贝 const 对象时引用计数不是会变吗？
- 核心：const 只管**句柄自身**（`base_` 指针成员不可改写），不传给被管对象或控制块——ShyPtr 等价于 `T* const` 而非 `const T*`。拷贝 const ShyPtr 合法：拷贝不改自身成员，只是把**堆上控制块**计数 +1，计数变化发生在指针所指的堆对象而非 const 对象内。
- 由此推 API 形状：拷贝构造/赋值必须收 `const ShyPtr&`（原实现 `ShyPtr(ShyPtr&)` 导致 const 左值拷不进去，即 todo 根源）；`get()/use_count()/operator->` 等只读观察者标 const；`reset()/operator=` 改写 base_ 故必须非 const；移动构造收 `ShyPtr&&`（偷指针并置空本身就是改写）。demo1 的 `print(ShyPtr<Base> const& sp)` 能过依赖观察者标 const。
(2026-09-04 17:36)
### Q: 那些 const/assign/lock bug 之前就在，为什么 cppreference demo 跑通了却没测出来？
- cppreference 的 example 是**正向功能演示**，不是测试：只走 happy path、无断言、不看析构次数/计数平衡/泄漏。demo1（多线程共享）从没做过赋值、自赋值、const 对象拷贝、reset 后单所有者回收、WeakPtr::lock——所以每个 bug 的触发场景它都没覆盖（赋值漏 return 需用到返回值；lock 悬垂需调用 lock；reset 泄漏需"单所有者 reset 后不再析构"，demo 里 reset 时线程还持有着）。
- 放大因素一：**模板成员惰性实例化**——operator= 漏 return、lock 悬垂这类错，只要该成员从没被调用，编译器就不为其生成代码，连 -Wall -Wreturn-type 都沉默，坏代码根本没被编出来。放大因素二：demo 只看打印输出，没有 alive 计数 + sanitizer 这类校验。
- 教训：抓这种 bug 靠**对抗性单测**——显式调用每个特殊成员（拷贝/移动构造、拷贝/移动赋值、自赋值、const 拷贝、重复 reset、weak lock），配 alive 计数 + UBSan/ASan，而不是跑功能演示。这也是为什么修完要补 probe_const 这种穷举 probe。
(2026-09-04 17:45)
### Q: 官方的 std::shared_ptr 控制块里 deleter 到底怎么存的？（libc++ 源码）
- deleter 不是"函数指针 + 无状态 lambda"能装的（运行期值无状态装不下、捕获后又不能转函数指针）——它是**一个有值的对象**，被**按值存进具体控制块**，控制块按 (指针类型, deleter, allocator) **模板化**，靠继承做类型擦除。
- libc++ 结构（本机 __memory/shared_count.h + __memory/shared_ptr.h）：基类 `__shared_count` 只放强计数 + 纯虚 `__on_zero_shared()`；`__shared_weak_count` 再加弱计数 + 纯虚 `__on_zero_shared_weak()`；具体块 `__shared_ptr_pointer<_Tp,_Dp,_Alloc> : __shared_weak_count`，成员 `_LIBCPP_COMPRESSED_TRIPLE(__ptr_, __deleter_, __alloc_)`（EBO 压成员）。shared_ptr 只持 `__shared_count*`，故 Y/D/Alloc 进不了基类类型。
- 生命周期：强计数归零 `__release_shared()` 调虚 `__on_zero_shared()` → `__deleter_(__ptr_); __deleter_.~_Dp();`（用完即析构 deleter，配合 allocator 走 placement-delete 释放块，不二次调 dtor）；弱也归零 → `__on_zero_shared_weak()` 用 allocator 释放内存。强计数用原子 decrement，等于 -1 触发（官方计数从 0 往 -1 走，use_count = 值+1）。
- 我们的简化版对应：`ShyPtrBase<T>`（保留 ptr_/get，纯虚 dispose）↔ 官方计数基类；`ShyPtrCtl<T,Y,D>`（按值存 D del_ + Y* raw_，dispose 调 del_(raw_)）↔ __shared_ptr_pointer。没照抄的两点：官方把指针放派生、shared_ptr 另存一份（为 alias/数组），我们用 delete base_ 而非 allocator placement-delete，所以不提前手动析构 deleter、留给派生 dtor。
(2026-09-04 17:59)
<!-- 以下继续记录 -->
