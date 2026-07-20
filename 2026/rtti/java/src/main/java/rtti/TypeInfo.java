package rtti;

import java.util.Objects;

public final class TypeInfo implements Comparable<TypeInfo> {

    private int id;
    private int high;
    private final String name;
    private final TypeInfo parent;
    private final Class<?> javaClass;

    TypeInfo(int id, int high, String name, TypeInfo parent, Class<?> javaClass) {
        this.id = id;
        this.high = high;
        this.name = Objects.requireNonNull(name);
        this.parent = parent;
        this.javaClass = javaClass;
    }

    void setId(int id) { this.id = id; }
    void setHigh(int high) { this.high = high; }

    public int id() { return id; }
    public int high() { return high; }
    public String name() { return name; }
    public TypeInfo parent() { return parent; }
    public Class<?> javaClass() { return javaClass; }

    public boolean isAssignableFrom(int typeId) {
        return typeId >= id && typeId <= high;
    }

    public boolean isAssignableFrom(TypeInfo other) {
        return other != null && isAssignableFrom(other.id);
    }

    public boolean isAssignableTo(TypeInfo other) {
        return other != null && other.isAssignableFrom(id);
    }

    public String hierarchyPath() {
        StringBuilder sb = new StringBuilder(name);
        for (TypeInfo p = parent; p != null; p = p.parent()) {
            sb.insert(0, " -> ").insert(0, p.name());
        }
        return sb.toString();
    }

    @Override
    public int compareTo(TypeInfo o) {
        return name.compareTo(o.name);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TypeInfo that)) return false;
        return name.equals(that.name) && id == that.id;
    }

    @Override
    public int hashCode() {
        return name.hashCode();
    }

    @Override
    public String toString() {
        return name + "[" + id + "," + high + "]";
    }
}
