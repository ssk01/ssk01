package rtti;

public abstract class RttiObject {

    protected final int __rttiTypeId;

    protected RttiObject(TypeInfo type) {
        this.__rttiTypeId = type.id();
    }

    protected RttiObject(int typeId) {
        this.__rttiTypeId = typeId;
    }

    public int typeId() {
        return __rttiTypeId;
    }

    public TypeInfo typeOf() {
        return TypeRegistry.get().lookup(__rttiTypeId);
    }

    public boolean isInstance(TypeInfo type) {
        return type != null && type.isAssignableFrom(__rttiTypeId);
    }

    public boolean isInstance(String typeName) {
        TypeInfo type = TypeRegistry.get().lookup(typeName);
        return isInstance(type);
    }

    @SuppressWarnings("unchecked")
    public <T extends RttiObject> T cast(Class<T> cls) {
        TypeInfo target = TypeRegistry.get().lookup(cls);
        if (target == null) {
            throw new ClassCastException("Unknown RTTI type for class: " + cls.getName());
        }
        if (!isInstance(target)) {
            throw new ClassCastException(
                    typeOf().name() + " cannot be cast to " + target.name());
        }
        return (T) this;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{type=" + typeOf().name() + "}";
    }
}
