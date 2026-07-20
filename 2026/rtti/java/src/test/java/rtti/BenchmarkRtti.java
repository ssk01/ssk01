package rtti;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class BenchmarkRtti {

    private TypeInfo entityType;
    private TypeInfo playerType;
    private TypeInfo enemyType;
    private TypeInfo orcType;
    private TypeInfo dragonType;

    private RttiObject[] objects;
    private int index;

    @Setup
    public void setup() {
        TypeRegistry reg = TypeRegistry.get();

        entityType = reg.define("Entity", null, Entity.class);
        playerType = reg.define("Player", entityType, Player.class);
        enemyType = reg.define("Enemy", entityType, Enemy.class);
        orcType = reg.define("Orc", enemyType, Orc.class);
        dragonType = reg.define("Dragon", enemyType, Dragon.class);

        reg.seal();

        objects = new RttiObject[]{
                new Player(playerType),
                new Orc(orcType),
                new Dragon(dragonType),
                new Player(playerType),
                new Orc(orcType),
                new Dragon(dragonType),
                new Player(playerType),
                new Orc(orcType),
        };
    }

    @Benchmark
    public boolean rttiIsInstance() {
        RttiObject obj = objects[index];
        index = (index + 1) % objects.length;
        return obj.isInstance(enemyType);
    }

    @Benchmark
    public boolean rttiIntervalRaw() {
        RttiObject obj = objects[index];
        index = (index + 1) % objects.length;
        int id = obj.typeId();
        return id >= enemyType.id() && id <= enemyType.high();
    }

    @Benchmark
    @SuppressWarnings("all")
    public boolean javaInstanceof() {
        RttiObject obj = objects[index];
        index = (index + 1) % objects.length;
        return obj instanceof Enemy;
    }

    @Benchmark
    public RttiObject rttiCast() {
        RttiObject obj = objects[index];
        index = (index + 1) % objects.length;
        return obj.cast(Enemy.class);
    }

    public static void main(String[] args) throws Exception {
        Options opt = new OptionsBuilder()
                .include(BenchmarkRtti.class.getSimpleName())
                .build();
        new Runner(opt).run();
    }

    static class Entity extends RttiObject { Entity() { super(0); } }
    interface EnemyMarker {}
    static class Player extends Entity implements EnemyMarker { Player(TypeInfo t) { super(t); } }
    static class Enemy extends Entity implements EnemyMarker { Enemy(TypeInfo t) { super(t); } }
    static class Orc extends Enemy { Orc(TypeInfo t) { super(t); } }
    static class Dragon extends Enemy { Dragon(TypeInfo t) { super(t); } }
}
