package rtti;

public final class Rtti {

    private Rtti() {}

    public static TypeInfo typeOf(RttiObject obj) {
        return obj == null ? null : obj.typeOf();
    }

    public static boolean isInstance(RttiObject obj, TypeInfo type) {
        return obj != null && obj.isInstance(type);
    }

    public static boolean isInstance(RttiObject obj, String typeName) {
        return obj != null && obj.isInstance(typeName);
    }

    public static boolean isAssignableFrom(TypeInfo superType, TypeInfo subType) {
        return superType != null && superType.isAssignableFrom(subType);
    }

    public static <T extends RttiObject> T cast(RttiObject obj, Class<T> cls) {
        return obj == null ? null : obj.cast(cls);
    }
}
