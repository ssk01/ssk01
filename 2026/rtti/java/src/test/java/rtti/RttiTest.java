package rtti;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RttiTest {

    static TypeInfo entityType;
    static TypeInfo playerType;
    static TypeInfo enemyType;
    static TypeInfo orcType;
    static TypeInfo dragonType;

    @BeforeAll
    static void setup() {
        TypeRegistry reg = TypeRegistry.get();

        entityType = reg.define("Entity", null, Entity.class);
        playerType = reg.define("Player", entityType, Player.class);
        enemyType = reg.define("Enemy", entityType, Enemy.class);
        orcType = reg.define("Orc", enemyType, Orc.class);
        dragonType = reg.define("Dragon", enemyType, Dragon.class);

        reg.seal();
    }

    @Test
    void testIntervalAssignment() {
        assertEquals(1, entityType.id());
        assertEquals(5, entityType.high());

        assertEquals(2, playerType.id());
        assertEquals(2, playerType.high());

        assertEquals(3, enemyType.id());
        assertEquals(5, enemyType.high());

        assertEquals(4, orcType.id());
        assertEquals(4, orcType.high());

        assertEquals(5, dragonType.id());
        assertEquals(5, dragonType.high());
    }

    @Test
    void testIsAssignableFrom() {
        assertTrue(entityType.isAssignableFrom(playerType));
        assertTrue(entityType.isAssignableFrom(enemyType));
        assertTrue(entityType.isAssignableFrom(orcType));
        assertTrue(entityType.isAssignableFrom(dragonType));

        assertFalse(playerType.isAssignableFrom(entityType));
        assertFalse(playerType.isAssignableFrom(enemyType));
        assertFalse(playerType.isAssignableFrom(orcType));

        assertTrue(enemyType.isAssignableFrom(orcType));
        assertTrue(enemyType.isAssignableFrom(dragonType));
        assertFalse(enemyType.isAssignableFrom(entityType));
        assertFalse(enemyType.isAssignableFrom(playerType));
    }

    @Test
    void testObjectTypeCheck() {
        RttiObject orc = new Orc(orcType);

        assertTrue(orc.isInstance(entityType));
        assertFalse(orc.isInstance(playerType));
        assertTrue(orc.isInstance(enemyType));
        assertTrue(orc.isInstance(orcType));
        assertFalse(orc.isInstance(dragonType));

        assertEquals("Orc", orc.typeOf().name());
    }

    @Test
    void testCastSuccess() {
        Enemy enemy = new Orc(orcType);

        Orc orc = enemy.cast(Orc.class);
        assertNotNull(orc);
    }

    @Test
    void testCastFail() {
        Enemy enemy = new Dragon(dragonType);

        assertThrows(ClassCastException.class, () -> enemy.cast(Orc.class));
    }

    @Test
    void testLookup() {
        assertEquals("Entity", TypeRegistry.get().lookup(1).name());
        assertEquals("Orc", TypeRegistry.get().lookup(4).name());
        assertEquals("Dragon", TypeRegistry.get().lookup("Dragon").name());
        assertEquals(Entity.class, TypeRegistry.get().lookup(Entity.class).javaClass());
    }

    @Test
    void testRttiUtil() {
        Player player = new Player(playerType);
        assertTrue(Rtti.isInstance(player, entityType));
        assertFalse(Rtti.isInstance(player, enemyType));
        assertEquals("Player", Rtti.typeOf(player).name());
    }

    @Test
    void testHugeHierarchy() {
        TypeRegistry reg = new TypeRegistry();

        TypeInfo root = reg.define("Root", null);
        for (int i = 0; i < 1000; i++) {
            reg.define("Child" + i, root);
        }
        reg.seal();

        TypeInfo child0 = reg.lookup("Child0");
        assertEquals(2, child0.id());
        assertEquals(2, child0.high());
        assertTrue(root.isAssignableFrom(child0));
        assertFalse(child0.isAssignableFrom(root));
    }

    // ---- Test model classes (no static TYPE fields) ----

    static class Entity extends RttiObject {
        Entity() { super(0); }
    }

    static class Player extends Entity {
        Player(TypeInfo t) { super(t); }
    }

    static class Enemy extends Entity {
        Enemy(TypeInfo t) { super(t); }
    }

    static class Orc extends Enemy {
        Orc(TypeInfo t) { super(t); }
    }

    static class Dragon extends Enemy {
        Dragon(TypeInfo t) { super(t); }
    }
}
