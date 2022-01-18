package kernel;

import rescuecore2.connection.Connection;
import rescuecore2.connection.ConnectionListener;
import rescuecore2.messages.Message;
import rescuecore2.messages.Command;
import rescuecore2.messages.control.SKUpdate;
import rescuecore2.messages.control.SKUpdateWithReward;
import rescuecore2.messages.control.KSUpdate;
import rescuecore2.messages.control.KSCommands;
import rescuecore2.messages.control.EntityIDRequest;
import rescuecore2.messages.control.EntityIDResponse;
import rescuecore2.worldmodel.ChangeSet;
import rescuecore2.worldmodel.RewardSet;
import rescuecore2.worldmodel.EntityID;
import rescuecore2.log.Logger;

import java.util.Collection;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * This class is the kernel interface to a simulator.
 */
public class SimulatorProxy extends AbstractKernelComponent {
    private Map<Integer, ChangeSet> updates;
    private Map<Integer, RewardSet> updates_reward;
    private int id;
    private EntityIDGenerator idGenerator;

    /**
     * Construct a new simulator.
     * 
     * @param name The name of the simulator.
     * @param id   The ID of the simulator.
     * @param c    The connection this simulator is using.
     */
    public SimulatorProxy(String name, int id, Connection c) {
        super(name, c);
        this.id = id;
        updates = new HashMap<Integer, ChangeSet>();
        updates_reward = new HashMap<Integer, RewardSet>();
        c.addConnectionListener(new SimulatorConnectionListener());
    }

    public void reset() {
        updates = new HashMap<Integer, ChangeSet>();
        updates_reward = new HashMap<Integer, RewardSet>();
    }

    /**
     * Get updates from this simulator. This method may block until updates are
     * available.
     * 
     * @param time The timestep to get updates for.
     * @return A ChangeSet representing the updates from this simulator.
     * @throws InterruptedException If this thread is interrupted while waiting for
     *                              updates.
     */
    public ChangeSet getUpdates(int time) throws InterruptedException {
        ChangeSet result = null;
        synchronized (updates) {
            while (result == null) {
                result = updates.remove(0);
                if (result == null) {
                    updates.wait(1000);
                }
            }
            // updates.clear();
        }

        return result;
    }

    public RewardSet getUpdatesReward(int time) throws InterruptedException {
        RewardSet result = null;
        synchronized (updates_reward) {
            while (result == null) {
                result = updates_reward.remove(0);
                if (result == null) {
                    updates_reward.wait(1000);
                }
            }
            // updates_reward.clear();

        }

        return result;
    }

    /**
     * Send an update message to this simulator.
     * 
     * @param time   The simulation time.
     * @param update The updated entities.
     */
    public void sendUpdate(int time, ChangeSet update) {
        send(new KSUpdate(id, time, update));
    }

    /**
     * Send a set of agent commands to this simulator.
     * 
     * @param time     The current time.
     * @param commands The agent commands to send.
     */
    public void sendAgentCommands(int time, Collection<? extends Command> commands) {
        send(new KSCommands(id, time, commands));
    }

    @Override
    public String toString() {
        return getName() + " (" + id + "): " + getConnection().toString();
    }

    /**
     * Set the EntityIDGenerator.
     * 
     * @param generator The new EntityIDGenerator.
     */
    public void setEntityIDGenerator(EntityIDGenerator generator) {
        idGenerator = generator;
    }

    /**
     * Register an update from the simulator.
     * 
     * @param time    The timestep of the update.
     * @param changes The set of changes.
     */
    protected void updateReceived(int time, ChangeSet changes) {
        synchronized (updates) {
            ChangeSet c = updates.get(time);
            if (c == null) {
                c = new ChangeSet();
                updates.put(time, c);
            }
            c.merge(changes);
            updates.notifyAll();
        }
    }

    protected void updateReceived(int time, ChangeSet changes, RewardSet rewards) {
        synchronized (updates_reward) {
            RewardSet r = updates_reward.get(0);
            if (r == null) {
                r = new RewardSet();
                updates_reward.put(0, r);
            }
            r.merge(rewards);
            updates_reward.notifyAll();
        }
        synchronized (updates) {
            ChangeSet c = updates.get(0);
            if (c == null) {
                c = new ChangeSet();
                updates.put(0, c);
            }
            c.merge(changes);
            updates.notifyAll();
        }
    }

    private class SimulatorConnectionListener implements ConnectionListener {
        @Override
        public void messageReceived(Connection connection, Message msg) {
            Logger.pushLogContext(Kernel.KERNEL_LOG_CONTEXT);
            try {
                /*
                 * if (msg instanceof SKUpdate) { Logger.debug("SKUpdate");
                 * 
                 * SKUpdate update = (SKUpdate)msg; if (update.getSimulatorID() == id) {
                 * updateReceived(update.getTime(), update.getChangeSet()); } }
                 */
                if (msg instanceof SKUpdateWithReward) {
                    Logger.debug("SKUpdateWithReward");

                    SKUpdateWithReward update = (SKUpdateWithReward) msg;
                    if (update.getSimulatorID() == id) {
                        updateReceived(update.getTime(), update.getChangeSet(), update.getRewardSet());
                    }
                }
                if (msg instanceof EntityIDRequest) {
                    EntityIDRequest req = (EntityIDRequest) msg;
                    Logger.debug("Simulator proxy " + id + " received entity ID request: " + msg);
                    if (req.getSimulatorID() == id) {
                        int requestID = req.getRequestID();
                        int count = req.getCount();
                        List<EntityID> result = new ArrayList<EntityID>(count);
                        for (int i = 0; i < count; ++i) {
                            result.add(idGenerator.generateID());
                        }
                        Logger.debug("Simulator proxy " + id + " sending new IDs: " + result);
                        send(new EntityIDResponse(id, requestID, result));
                    }
                }
            } finally {
                Logger.popLogContext();
            }
        }
    }

    @Override
    public int hashCode() {
        return Integer.hashCode(id);
    }
}
