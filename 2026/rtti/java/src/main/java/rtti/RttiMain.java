package rtti;

public class RttiMain {
    static TypeInfo entityType;
    static TypeInfo playerType;
    static TypeInfo enemyType;
    static TypeInfo orcType;
    static TypeInfo dragonType;

    static int passed = 0;
    static int failed = 0;

    public static void main(String[] args) {
        TypeRegistry reg = TypeRegistry.get();

        entityType = reg.define("Entity", null, Entity.class);
        playerType = reg.define("Player", entityType, Player.class);
        enemyType = reg.define("Enemy", entityType, Enemy.class);
        orcType = reg.define("Orc", enemyType, Orc.class);
        dragonType = reg.define("Dragon", enemyType, Dragon.class);

        reg.seal();
        System.out.println("Registry: " + reg);
        System.out.println();

        test("interval assignment", () -> {
            check(entityType.id() == 1 && entityType.high() == 5, "Entity [1,5]");
            check(enemyType.id() == 2 && enemyType.high() == 4, "Enemy [2,4]");
            check(dragonType.id() == 3 && dragonType.high() == 3, "Dragon [3,3]");
            check(orcType.id() == 4 && orcType.high() == 4, "Orc [4,4]");
            check(playerType.id() == 5 && playerType.high() == 5, "Player [5,5]");
        });

        test("isAssignableFrom", () -> {
            check(entityType.isAssignableFrom(playerType), "Entity -> Player");
            check(entityType.isAssignableFrom(orcType), "Entity -> Orc");
            check(!playerType.isAssignableFrom(entityType), "!Player -> Entity");
            check(!playerType.isAssignableFrom(orcType), "!Player -> Orc");
            check(enemyType.isAssignableFrom(orcType), "Enemy -> Orc");
            check(enemyType.isAssignableFrom(dragonType), "Enemy -> Dragon");
            check(!enemyType.isAssignableFrom(playerType), "!Enemy -> Player");
            check(!enemyType.isAssignableFrom(entityType), "!Enemy -> Entity");
        });

        test("object type check", () -> {
            Orc orc = new Orc(orcType);
            check(orc.isInstance(entityType), "orc is Entity");
            check(!orc.isInstance(playerType), "orc is not Player");
            check(orc.isInstance(enemyType), "orc is Enemy");
            check(orc.isInstance(orcType), "orc is Orc");
            check(!orc.isInstance(dragonType), "orc is not Dragon");
            check("Orc".equals(orc.typeOf().name()), "typeOf == Orc");
        });

        test("cast success", () -> {
            Enemy enemy = new Orc(orcType);
            Orc orc = enemy.cast(Orc.class);
            check(orc != null, "cast succeeded");
        });

        test("cast fail", () -> {
            Enemy dragon = new Dragon(dragonType);
            try {
                dragon.cast(Orc.class);
                check(false, "should have thrown");
            } catch (ClassCastException e) {
                check(true, "caught ClassCastException");
            }
        });

        test("lookup by id/name/class", () -> {
            check("Entity".equals(TypeRegistry.get().lookup(1).name()), "id→Entity");
            check("Orc".equals(TypeRegistry.get().lookup(4).name()), "id→Orc");
            check("Dragon".equals(TypeRegistry.get().lookup("Dragon").name()), "name→Dragon");
            check(Entity.class == TypeRegistry.get().lookup(Entity.class).javaClass(), "class lookup");
        });

        test("Rtti utility", () -> {
            Player p = new Player(playerType);
            check(Rtti.isInstance(p, entityType), "isInstance yes");
            check(!Rtti.isInstance(p, enemyType), "isInstance no");
            check("Player".equals(Rtti.typeOf(p).name()), "typeOf");
        });

        test("large hierarchy", () -> {
            TypeRegistry reg2 = new TypeRegistry();
            TypeInfo root = reg2.define("Root", null);
            for (int i = 0; i < 500; i++) {
                reg2.define("Child" + i, root);
            }
            reg2.seal();
            TypeInfo child = reg2.lookup("Child0");
            check(child.id() == 2 && child.high() == 2, "Child0 [2,2]");
            check(root.isAssignableFrom(child), "Root -> Child0");
            check(!child.isAssignableFrom(root), "!Child0 -> Root");
        });

        System.out.println("\n" + passed + " passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    static void test(String name, Runnable r) {
        try {
            r.run();
        } catch (Exception e) {
            System.out.println("  FAIL " + name + ": " + e.getMessage());
            failed++;
            return;
        }
        System.out.println("  PASS " + name);
        passed++;
    }

    static void check(boolean cond, String msg) {
        if (!cond) throw new AssertionError(msg);
    }

    static class Entity extends RttiObject {
        Entity() { super(0); }
        Entity(TypeInfo t) { super(t); }
    }
    static class Player extends Entity { Player(TypeInfo t) { super(t); } }
    static class Enemy extends Entity { Enemy(TypeInfo t) { super(t); } }
    static class Orc extends Enemy  { Orc(TypeInfo t) { super(t); } }
    static class Dragon extends Enemy { Dragon(TypeInfo t) { super(t); } }
}
