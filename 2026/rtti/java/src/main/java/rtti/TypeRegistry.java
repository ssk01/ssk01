package rtti;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public final class TypeRegistry {

    private static final TypeRegistry INSTANCE = new TypeRegistry();

    private final Map<String, TypeInfo> byName = new ConcurrentHashMap<>();
    private final Map<Class<?>, TypeInfo> byClass = new ConcurrentHashMap<>();
    private volatile TypeInfo[] byId = new TypeInfo[0];
    private volatile boolean sealed = false;

    public TypeRegistry() {}

    public static TypeRegistry get() { return INSTANCE; }

    public synchronized TypeInfo define(String name, TypeInfo parent) {
        return define(name, parent, null);
    }

    public synchronized TypeInfo define(String name, TypeInfo parent, Class<?> javaClass) {
        if (sealed) throw new IllegalStateException("Registry already sealed");
        if (byName.containsKey(name)) throw new IllegalArgumentException("Duplicate type: " + name);

        TypeInfo type = new TypeInfo(0, 0, name, parent, javaClass);
        byName.put(name, type);
        if (javaClass != null) {
            byClass.put(javaClass, type);
        }
        return type;
    }

    public synchronized void seal() {
        if (sealed) return;

        List<TypeInfo> roots = byName.values().stream()
                .filter(t -> t.parent() == null)
                .sorted(Comparator.comparing(TypeInfo::name))
                .collect(Collectors.toList());

        List<TypeInfo> ordered = new ArrayList<>();
        AtomicInteger counter = new AtomicInteger(1);

        for (TypeInfo root : roots) {
            dfsAssign(root, ordered, counter);
        }

        int maxId = counter.get() - 1;
        byId = new TypeInfo[maxId + 1];
        for (TypeInfo t : ordered) {
            byId[t.id()] = t;
        }

        sealed = true;
    }

    private int dfsAssign(TypeInfo node, List<TypeInfo> ordered, AtomicInteger counter) {
        int id = counter.getAndIncrement();
        int high = id;

        node.setId(id);
        node.setHigh(high);

        List<TypeInfo> children = byName.values().stream()
                .filter(t -> t.parent() != null && t.parent().name().equals(node.name()))
                .sorted(Comparator.comparing(TypeInfo::name))
                .collect(Collectors.toList());

        for (TypeInfo child : children) {
            int childHigh = dfsAssign(child, ordered, counter);
            high = childHigh;
            node.setHigh(high);
        }

        ordered.add(node);
        return high;
    }

    public TypeInfo lookup(int typeId) {
        TypeInfo[] snapshot = byId;
        if (typeId < 0 || typeId >= snapshot.length) return null;
        return snapshot[typeId];
    }

    public TypeInfo lookup(String name) {
        return byName.get(name);
    }

    public TypeInfo lookup(Class<?> clazz) {
        return byClass.get(clazz);
    }

    public boolean isAssignableFrom(TypeInfo superType, int typeId) {
        return superType != null && superType.isAssignableFrom(typeId);
    }

    public boolean isSealed() { return sealed; }

    public List<TypeInfo> allTypes() {
        return Collections.unmodifiableList(Arrays.asList(byId).subList(1, byId.length));
    }

    @Override
    public String toString() {
        if (!sealed) return "TypeRegistry[unsealed, " + byName.size() + " types]";
        StringBuilder sb = new StringBuilder("TypeRegistry[");
        for (int i = 1; i < byId.length; i++) {
            if (i > 1) sb.append(", ");
            sb.append(byId[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}
