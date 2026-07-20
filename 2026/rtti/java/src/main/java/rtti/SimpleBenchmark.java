package rtti;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SimpleBenchmark {

    static final int N_OBJECTS = 100_000;
    static final int N_ITERS = 100;

    public static void main(String[] args) {
        TypeRegistry reg = new TypeRegistry();

        TypeInfo entityType = reg.define("Entity", null, Entity.class);
        TypeInfo enemyType  = reg.define("Enemy", entityType, Enemy.class);
        TypeInfo playerType = reg.define("Player", entityType, Player.class);
        TypeInfo orcType    = reg.define("Orc", enemyType, Orc.class);
        TypeInfo dragonType = reg.define("Dragon", enemyType, Dragon.class);
        reg.seal();

        Random rng = new Random(42);
        List<RttiObject> objects = new ArrayList<>(N_OBJECTS);
        for (int i = 0; i < N_OBJECTS; i++) {
            switch (rng.nextInt(3)) {
                case 0 -> objects.add(new Player(playerType));
                case 1 -> objects.add(new Orc(orcType));
                case 2 -> objects.add(new Dragon(dragonType));
            }
        }

        // ---- RTTI interval check ----
        {
            int count = 0;
            long start = System.nanoTime();
            for (int iter = 0; iter < N_ITERS; iter++) {
                for (RttiObject obj : objects) {
                    if (obj.isInstance(enemyType)) count++;
                }
            }
            long elapsed = System.nanoTime() - start;
            double nsPer = (double) elapsed / (N_OBJECTS * N_ITERS);
            System.out.printf("[RTTI interval]  count=%d  avg %.2f ns/check%n", count, nsPer);
        }

        // ---- RTTI raw int ----
        {
            int count = 0;
            int low = enemyType.id(), high = enemyType.high();
            long start = System.nanoTime();
            for (int iter = 0; iter < N_ITERS; iter++) {
                for (RttiObject obj : objects) {
                    int id = obj.typeId();
                    if (id >= low && id <= high) count++;
                }
            }
            long elapsed = System.nanoTime() - start;
            double nsPer = (double) elapsed / (N_OBJECTS * N_ITERS);
            System.out.printf("[RTTI raw int]   count=%d  avg %.2f ns/check%n", count, nsPer);
        }

        // ---- Java instanceof ----
        {
            int count = 0;
            long start = System.nanoTime();
            for (int iter = 0; iter < N_ITERS; iter++) {
                for (RttiObject obj : objects) {
                    if (obj instanceof Enemy) count++;
                }
            }
            long elapsed = System.nanoTime() - start;
            double nsPer = (double) elapsed / (N_OBJECTS * N_ITERS);
            System.out.printf("[java instanceof] count=%d  avg %.2f ns/check%n", count, nsPer);
        }

        // ---- noop baseline ----
        {
            int count = 0;
            long start = System.nanoTime();
            for (int iter = 0; iter < N_ITERS; iter++) {
                for (RttiObject obj : objects) {
                    if (obj.typeId() > 0) count++;
                }
            }
            long elapsed = System.nanoTime() - start;
            double nsPer = (double) elapsed / (N_OBJECTS * N_ITERS);
            System.out.printf("[noop baseline]  count=%d  avg %.2f ns/check%n", count, nsPer);
        }
    }

    static class Entity extends RttiObject {
        Entity() { super(0); }
        Entity(int typeId) { super(typeId); }
        Entity(TypeInfo t) { super(t); }
    }

    static class Player extends Entity { Player(TypeInfo t) { super(t); } }

    static class Enemy extends Entity {
        Enemy() { super(0); }
        Enemy(TypeInfo t) { super(t); }
    }

    static class Orc extends Enemy { Orc(TypeInfo t) { super(t); } }

    static class Dragon extends Enemy { Dragon(TypeInfo t) { super(t); } }
}
