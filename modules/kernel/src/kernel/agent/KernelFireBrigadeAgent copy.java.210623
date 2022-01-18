package kernel.agent;

import static rescuecore2.misc.Handy.objectsToIDs;

import java.util.List;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Map;
import java.util.HashSet;
import java.util.Set;
import java.util.Random;

import java.io.File;

import rescuecore2.config.Config;
import rescuecore2.worldmodel.Entity;
import rescuecore2.worldmodel.EntityID;
import rescuecore2.worldmodel.WorldModel;
import rescuecore2.worldmodel.DefaultWorldModel;
import rescuecore2.standard.entities.StandardWorldModel;
import rescuecore2.standard.messages.AKClear;
import rescuecore2.standard.messages.AKClearArea;
import rescuecore2.standard.messages.AKExtinguish;
import rescuecore2.standard.messages.AKLoad;
import rescuecore2.standard.messages.AKMove;
import rescuecore2.standard.messages.AKRescue;
import rescuecore2.standard.messages.AKRest;
import rescuecore2.standard.messages.AKSay;
import rescuecore2.standard.messages.AKSpeak;
import rescuecore2.standard.messages.AKSubscribe;
import rescuecore2.standard.messages.AKTell;
import rescuecore2.standard.messages.AKUnload;
import rescuecore2.messages.Command;
import rescuecore2.log.Logger;

import rescuecore2.standard.entities.StandardEntity;
import rescuecore2.standard.entities.StandardEntityURN;
import rescuecore2.standard.entities.Building;
import rescuecore2.standard.entities.Refuge;
import rescuecore2.standard.entities.FireBrigade;
import rescuecore2.standard.entities.Human;
import rescuecore2.standard.entities.Road;

import sample.DistanceSorter;
import sample.SampleSearch;

public class KernelFireBrigadeAgent implements Comparable<KernelFireBrigadeAgent> {

    private static final int RANDOM_WALK_LENGTH = 50;
    private Random random;
    private static final String MAX_WATER_KEY = "fire.tank.maximum";
    private static final String MAX_DISTANCE_KEY = "fire.extinguish.max-distance";
    private static final String MAX_POWER_KEY = "fire.extinguish.max-sum";

    private int maxWater;
    private int maxDistance;
    private int maxPower;

    private EntityID entityID;
    private StandardWorldModel model;
    private SampleSearch search;

    protected List<EntityID> buildingIDs;

    /**
     * Cache of road IDs.
     */
    protected List<EntityID> roadIDs;

    /**
     * Cache of refuge IDs.
     */
    protected List<EntityID> refugeIDs;

    private Map<EntityID, Set<EntityID>> neighbours;

    protected Config config;

    public KernelFireBrigadeAgent(StandardWorldModel m, EntityID id, Config c) {
        this.entityID = id;
        this.model = m;
        buildingIDs = new ArrayList<EntityID>();
        roadIDs = new ArrayList<EntityID>();
        refugeIDs = new ArrayList<EntityID>();
        for (StandardEntity next : model) {
            if (next instanceof Building) {
                buildingIDs.add(next.getID());
            }
            if (next instanceof Road) {
                roadIDs.add(next.getID());
            }
            if (next instanceof Refuge) {
                refugeIDs.add(next.getID());
            }
        }
        search = new SampleSearch(model);
        neighbours = search.getGraph();
        config = new Config(c);
        random = config.getRandom();
        maxWater = config.getIntValue(MAX_WATER_KEY);
        maxDistance = config.getIntValue(MAX_DISTANCE_KEY);
        maxPower = config.getIntValue(MAX_POWER_KEY);

    }

    public FireBrigade me() {
        if (entityID == null) {
            return null;
        }
        if (model == null) {
            return null;
        }
        return (FireBrigade) model.getEntity(entityID);
    }

    public Command think(int time, EntityID buildingID, String command) {
        long beforeTime = System.currentTimeMillis();
        // Collection<EntityID> dest = new ArrayList<EntityID>();
        // dest.add(new EntityID(1108));
        // List<EntityID> path = search.breadthFirstSearch(me().getPosition(), dest);
        // sendMove(time, path);

        FireBrigade me = me();
        // fire resonse algorithm
        List<EntityID> path = randomWalk();

        if (buildingID.getValue() == -1) {
            return sendMove(time, path);
        }

        // Can we extinguish any right now?
        if (model.getDistance(getID(), buildingID) <= maxDistance) {
            return (Command) sendExtinguish(time, buildingID, maxPower);
        }
        // Plan a path to a fire
        path = planPathToFire(buildingID);
        if (path != null) {
            return (Command) sendMove(time, path);
        }

        path = randomWalk();

        return sendMove(time, path);
    }

    private StandardEntity location() {
        FireBrigade me = me();
        if (me instanceof Human) {
            return ((Human) me).getPosition(model);
        }
        return me;
    }

    public EntityID getID() {
        return entityID;
    }

    /**
     * Send a rest command to the kernel.
     * 
     * @param time The current time.
     */
    protected AKRest sendRest(int time) {
        return new AKRest(getID(), time);
    }

    /**
     * Send a move command to the kernel.
     * 
     * @param time The current time.
     * @param path The path to send.
     */
    protected AKMove sendMove(int time, List<EntityID> path) {
        return new AKMove(getID(), time, path);
    }

    /**
     * Send a move command to the kernel.
     * 
     * @param time  The current time.
     * @param path  The path to send.
     * @param destX The destination X coordinate.
     * @param destY The destination Y coordinate.
     */
    protected AKMove sendMove(int time, List<EntityID> path, int destX, int destY) {
        return new AKMove(getID(), time, path, destX, destY);
    }

    /**
     * Send an extinguish command to the kernel.
     * 
     * @param time   The current time.
     * @param target The target building.
     * @param water  The amount of water to use.
     */
    protected AKExtinguish sendExtinguish(int time, EntityID target, int water) {
        return new AKExtinguish(getID(), time, target, water);
    }

    /**
     * Send a clear command to the kernel.
     * 
     * @param time   The current time.
     * @param target The target road.
     */
    protected AKClear sendClear(int time, EntityID target) {
        return new AKClear(getID(), time, target);
    }

    /**
     * Send a clear command to the kernel.
     * 
     * @param time  The current time.
     * @param destX The destination X coordinate to clear.
     * @param destY The destination Y coordinate to clear.
     */
    protected AKClearArea sendClear(int time, int destX, int destY) {
        return new AKClearArea(getID(), time, destX, destY);
    }

    /**
     * Send a rescue command to the kernel.
     * 
     * @param time   The current time.
     * @param target The target human.
     */
    protected AKRescue sendRescue(int time, EntityID target) {
        return new AKRescue(getID(), time, target);
    }

    /**
     * Send a load command to the kernel.
     * 
     * @param time   The current time.
     * @param target The target human.
     */
    protected AKLoad sendLoad(int time, EntityID target) {
        return new AKLoad(getID(), time, target);
    }

    /**
     * Send an unload command to the kernel.
     * 
     * @param time The current time.
     */
    protected AKUnload sendUnload(int time) {
        return new AKUnload(getID(), time);
    }

    /**
     * Send a speak command to the kernel.
     * 
     * @param time    The current time.
     * @param channel The channel to speak on.
     * @param data    The data to send.
     */
    protected AKSpeak sendSpeak(int time, int channel, byte[] data) {
        return new AKSpeak(getID(), time, channel, data);
    }

    /**
     * Send a subscribe command to the kernel.
     * 
     * @param time     The current time.
     * @param channels The channels to subscribe to.
     */
    protected AKSubscribe sendSubscribe(int time, int... channels) {
        return new AKSubscribe(getID(), time, channels);
    }

    /**
     * Send a say command to the kernel.
     * 
     * @param time The current time.
     * @param data The data to send.
     */
    protected AKSay sendSay(int time, byte[] data) {
        return new AKSay(getID(), time, data);
    }

    /**
     * Send a tell command to the kernel.
     * 
     * @param time The current time.
     * @param data The data to send.
     */
    protected AKTell sendTell(int time, byte[] data) {
        return new AKTell(getID(), time, data);
    }

    public Collection<EntityID> getBurningBuildings() {
        Collection<StandardEntity> e = model.getEntitiesOfType(StandardEntityURN.BUILDING);
        List<Building> result = new ArrayList<Building>();
        for (StandardEntity next : e) {
            if (next instanceof Building) {
                Building b = (Building) next;
                if (b.isOnFire()) {
                    result.add(b);
                }
            }
        }
        // Sort by distance
        Collections.sort(result, new DistanceSorter(location(), model));
        return objectsToIDs(result);
    }

    private List<EntityID> planPathToFire(EntityID target) {
        // Try to get to anything within maxDistance of the target
        Collection<StandardEntity> targets = model.getObjectsInRange(target, (int) (maxDistance));
        targets.removeIf(e -> model.getDistance(target, e.getID()) > maxDistance);

        if (targets.isEmpty()) {
            return null;
        }
        return search.breadthFirstSearch(me().getPosition(), objectsToIDs(targets));
    }

    private List<EntityID> randomWalk() {
        List<EntityID> result = new ArrayList<EntityID>(RANDOM_WALK_LENGTH);
        Set<EntityID> seen = new HashSet<EntityID>();
        EntityID current = ((Human) me()).getPosition();
        for (int i = 0; i < RANDOM_WALK_LENGTH; ++i) {
            result.add(current);
            seen.add(current);
            List<EntityID> possible = new ArrayList<EntityID>(neighbours.get(current));
            Collections.shuffle(possible, random);
            boolean found = false;
            for (EntityID next : possible) {
                if (seen.contains(next)) {
                    continue;
                }
                current = next;
                found = true;
                break;
            }
            if (!found) {
                // We reached a dead-end.
                break;
            }
        }
        return result;
    }

    @Override
    public int compareTo(KernelFireBrigadeAgent kfba) {
        if (this.entityID.getValue() > kfba.entityID.getValue()) {
            return 1;
        } else {
            return -1;
        }
    }

}
