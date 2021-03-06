package kernel;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import kernel.agent.KernelFireBrigadeAgent;
import rescuecore2.Constants;
import rescuecore2.Timestep;
import rescuecore2.config.Config;
import rescuecore2.log.ConfigRecord;
import rescuecore2.log.EndLogRecord;
import rescuecore2.log.FileLogWriter;
import rescuecore2.log.InitialConditionsRecord;
import rescuecore2.log.LogException;
import rescuecore2.log.LogWriter;
import rescuecore2.log.Logger;
import rescuecore2.log.StartLogRecord;
import rescuecore2.log.UpdatesRecord;
import rescuecore2.messages.Command;
import rescuecore2.score.ScoreFunction;

import rescuecore2.standard.score.BuildingDamageScoreFunction;
import rescuecore2.standard.entities.AmbulanceTeam;
import rescuecore2.standard.entities.Blockade;
import rescuecore2.standard.entities.Building;
import rescuecore2.standard.entities.Civilian;
import rescuecore2.standard.entities.FireBrigade;
import rescuecore2.standard.entities.PoliceForce;
import rescuecore2.standard.entities.Refuge;
import rescuecore2.standard.entities.Road;
import rescuecore2.standard.entities.StandardWorldModel;
import rescuecore2.standard.entities.StandardEntity;
import rescuecore2.standard.messages.AKExtinguish;
import rescuecore2.standard.messages.AKMove;
import kernel.ui.ScoreGraph;
import kernel.ui.ScoreTable;
import rescuecore2.worldmodel.RewardSet;
import rescuecore2.worldmodel.ChangeSet;
import rescuecore2.worldmodel.Entity;
import rescuecore2.worldmodel.EntityID;
import rescuecore2.worldmodel.WorldModel;

//import rescuecore2.misc.gui.ChangeSetComponent;

/**
 * The Robocup Rescue kernel.
 */
public class Kernel {
    /** The log context for kernel log messages. */
    public static final String KERNEL_LOG_CONTEXT = "kernel";

    private Config config;
    private Perception perception;
    private CommunicationModel communicationModel;
    private WorldModel<? extends Entity> worldModel;
    private ChangeSet initialWorld;
    private ChangeSet initialWorld_RLEnv;
    private LogWriter log;

    private Set<KernelListener> listeners;

    private Collection<AgentProxy> agents;
    private Collection<SimulatorProxy> sims;
    private Collection<ViewerProxy> viewers;
    private int time;
    private Timestep previousTimestep;

    private EntityIDGenerator idGenerator;
    private CommandFilter commandFilter;

    private TerminationCondition termination;
    private ScoreFunction score;
    private CommandCollector commandCollector;

    private boolean isFireSimStarted = false;
    private boolean isShutdown;
    private KernelStartupOptions options;

    // for comm with python socket clients.
    private ServerSocket serverSocket;
    private Socket acceptSocket;
    private PrintWriter output;
    private InputStream input;
    private Gson gson;
    private String str1;
    private String str2;
    private String str3;
    // make scoreMap
    private HashMap<String, Double> scoreMap;

    // make agentinkernel
    ArrayList<KernelFireBrigadeAgent> kernelFireBrigadeAgents;

    // Building damage score
    BuildingDamageScoreFunction buildingDamageScore;

    // private ChangeSetComponent simulatorChanges;
    private int lastStartIdx = 0;

    class ResultfromSims {
        public ChangeSet changeSet;
        public RewardSet rewardSet;

        public ResultfromSims(ChangeSet c, RewardSet r) {
            this.changeSet = c;
            this.rewardSet = r;
        }
    }

    /**
     * Construct a kernel.
     * 
     * @param config             The configuration to use.
     * @param perception         A perception calculator.
     * @param communicationModel A communication model.
     * @param worldModel         The world model.
     * @param idGenerator        An EntityIDGenerator.
     * @param commandFilter      An optional command filter. This may be null.
     * @param termination        The termination condition.
     * @param score              The score function.
     * @param collector          The CommandCollector to use.
     * @throws KernelException If there is a problem constructing the kernel.
     */

    public Kernel(Config config, Perception perception, CommunicationModel communicationModel,
            WorldModel<? extends Entity> worldModel, EntityIDGenerator idGenerator, CommandFilter commandFilter,
            TerminationCondition termination, ScoreFunction score, CommandCollector collector,
            KernelStartupOptions options) throws KernelException {

        gson = new Gson();
        this.initcomm();
        kernelFireBrigadeAgents = new ArrayList<KernelFireBrigadeAgent>();

        ArrayList<Building> buildings = new ArrayList<Building>();
        ArrayList<Road> roads = new ArrayList<Road>();
        ArrayList<Refuge> refuges = new ArrayList<Refuge>();
        ArrayList<FireBrigade> firebrigades = new ArrayList<FireBrigade>();
        ArrayList<AmbulanceTeam> ambulanceTeams = new ArrayList<AmbulanceTeam>();
        ArrayList<Blockade> blockades = new ArrayList<Blockade>();
        ArrayList<Civilian> civilians = new ArrayList<Civilian>();
        ArrayList<PoliceForce> policeForces = new ArrayList<PoliceForce>();
        scoreMap = new HashMap<String, Double>();

        for (Entity next : worldModel.getAllEntities()) {
            if (next instanceof Building) {
                buildings.add((Building) next);
            }
            if (next instanceof Road) {
                roads.add((Road) next);
            }
            if (next instanceof Refuge) {
                refuges.add((Refuge) next);
            }
            if (next instanceof FireBrigade) {
                firebrigades.add((FireBrigade) next);
                kernelFireBrigadeAgents
                        .add(new KernelFireBrigadeAgent((StandardWorldModel) worldModel, next.getID(), config));

            }
            if (next instanceof AmbulanceTeam) {
                ambulanceTeams.add((AmbulanceTeam) next);
            }
            if (next instanceof Blockade) {
                blockades.add((Blockade) next);
            }
            if (next instanceof Civilian) {
                civilians.add((Civilian) next);
            }
            if (next instanceof PoliceForce) {
                policeForces.add((PoliceForce) next);
            }
        }
        Collections.sort(kernelFireBrigadeAgents);

        HashMap<Integer, Integer> neighborCountMap = new HashMap<Integer, Integer>();
        for (StandardEntity target : ((StandardWorldModel) worldModel).getAllEntities()) {
            if (target instanceof Building) {
                int neighborsCount = 0;

                for (StandardEntity rangeEntity : ((StandardWorldModel) worldModel).getObjectsInRange(target, 50000)) {
                    if (rangeEntity instanceof Building) {
                        Building rangeBuilding = (Building) rangeEntity;
                        neighborsCount++;
                    }
                }
                neighborCountMap.put(target.getID().getValue(), neighborsCount);
                // System.out.println("neighborsCount ::::" + neighborsCount);
            }
        }
        str1 = gson.toJson(buildings);
        str2 = gson.toJson(firebrigades);
        str3 = gson.toJson(neighborCountMap);
        // output.println(str1+"#"+str2 + "@");
        output.println(str1 + "#" + str2 + "#" + str3 + "@");

        try {
            Logger.pushLogContext(KERNEL_LOG_CONTEXT);
            this.buildingDamageScore = new BuildingDamageScoreFunction();
            this.config = config;
            this.perception = perception;
            this.communicationModel = communicationModel;
            this.worldModel = worldModel;
            this.commandFilter = commandFilter;
            this.score = score;
            this.termination = termination;
            this.commandCollector = collector;
            this.idGenerator = idGenerator;
            this.options = options;
            // try {
            // this.initialWorld = ((DefaultWorldModel)this.worldModel).clone();
            // }catch(Exception e){
            // e.printStackTrace();
            // }
            this.initialWorld = new ChangeSet();
            this.initialWorld.addAll(this.worldModel.getAllEntities());
            this.initialWorld_RLEnv = new ChangeSet();
            this.initialWorld_RLEnv.addAll(buildings);
            this.initialWorld_RLEnv.addAll(firebrigades);

            listeners = new HashSet<KernelListener>();
            agents = new TreeSet<AgentProxy>(new Comparator<AgentProxy>() {
                @Override
                public int compare(AgentProxy o1, AgentProxy o2) {
                    return Integer.compare(o1.hashCode(), o2.hashCode());
                }
            });
            sims = new HashSet<SimulatorProxy>();
            viewers = new HashSet<ViewerProxy>();
            time = 0;
            try {
                String logName = config.getValue("kernel.logname");
                Logger.info("Logging to " + logName);
                File logFile = new File(logName);
                if (logFile.getParentFile().mkdirs()) {
                    Logger.info("Created log directory: " + logFile.getParentFile().getAbsolutePath());
                }
                if (logFile.createNewFile()) {
                    Logger.info("Created log file: " + logFile.getAbsolutePath());
                }
                log = new FileLogWriter(logFile);
                log.writeRecord(new StartLogRecord());
                log.writeRecord(new InitialConditionsRecord(worldModel));
                log.writeRecord(new ConfigRecord(config));
            } catch (IOException e) {
                throw new KernelException("Couldn't open log file for writing", e);
            } catch (LogException e) {
                throw new KernelException("Couldn't open log file for writing", e);
            }
            config.setValue(Constants.COMMUNICATION_MODEL_KEY, communicationModel.getClass().getName());
            config.setValue(Constants.PERCEPTION_KEY, perception.getClass().getName());

            // simulatorChanges = new ChangeSetComponent();

            // Initialise
            perception.initialise(config, worldModel);
            communicationModel.initialise(config, worldModel);
            commandFilter.initialise(config);
            score.initialise(worldModel, config);
            this.buildingDamageScore.initialise(worldModel, config);
            termination.initialise(config);
            commandCollector.initialise(config);

            isShutdown = false;

            Logger.info("Kernel initialised");
            Logger.info("Perception module: " + perception);
            Logger.info("Communication module: " + communicationModel);
            Logger.info("Command filter: " + commandFilter);
            Logger.info("Score function: " + score);
            Logger.info("Termination condition: " + termination);
            Logger.info("Command collector: " + collector);
        } finally {
            Logger.popLogContext();
        }
    }

    // public void reset(){
    // isShutdown = false;
    // }
    /**
     * Get the kernel's configuration.
     * 
     * @return The configuration.
     */

    public void initcomm() {
        try {
            this.serverSocket = new ServerSocket(9999);
            acceptSocket = serverSocket.accept();
            output = new PrintWriter(acceptSocket.getOutputStream(), true);
            input = acceptSocket.getInputStream();
            System.out.println("1234");
            System.out.println("abab");

        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public Config getConfig() {
        return config;
    }

    /**
     * Get a snapshot of the kernel's state.
     * 
     * @return A new KernelState snapshot.
     */
    public KernelState getState() {
        return new KernelState(getTime(), getWorldModel());
    }

    /**
     * Add an agent to the system.
     * 
     * @param agent The agent to add.
     */
    public void addAgent(AgentProxy agent) {
        synchronized (this) {
            agents.add(agent);
        }
        fireAgentAdded(agent);
    }

    /**
     * Remove an agent from the system.
     * 
     * @param agent The agent to remove.
     */
    public void removeAgent(AgentProxy agent) {
        synchronized (this) {
            agents.remove(agent);
        }
        fireAgentRemoved(agent);
    }

    /**
     * Get all agents in the system.
     * 
     * @return An unmodifiable view of all agents.
     */
    public Collection<AgentProxy> getAllAgents() {
        synchronized (this) {
            return Collections.unmodifiableCollection(agents);
        }
    }

    /**
     * Add a simulator to the system.
     * 
     * @param sim The simulator to add.
     */
    public void addSimulator(SimulatorProxy sim) {
        synchronized (this) {
            sims.add(sim);
            sim.setEntityIDGenerator(idGenerator);
        }
        fireSimulatorAdded(sim);
    }

    /**
     * Remove a simulator from the system.
     * 
     * @param sim The simulator to remove.
     */
    public void removeSimulator(SimulatorProxy sim) {
        synchronized (this) {
            sims.remove(sim);
        }
        fireSimulatorRemoved(sim);
    }

    /**
     * Get all simulators in the system.
     * 
     * @return An unmodifiable view of all simulators.
     */
    public Collection<SimulatorProxy> getAllSimulators() {
        synchronized (this) {
            return Collections.unmodifiableCollection(sims);
        }
    }

    /**
     * Add a viewer to the system.
     * 
     * @param viewer The viewer to add.
     */
    public void addViewer(ViewerProxy viewer) {
        synchronized (this) {
            viewers.add(viewer);
        }
        fireViewerAdded(viewer);
    }

    /**
     * Remove a viewer from the system.
     * 
     * @param viewer The viewer to remove.
     */
    public void removeViewer(ViewerProxy viewer) {
        synchronized (this) {
            viewers.remove(viewer);
        }
        fireViewerRemoved(viewer);
    }

    /**
     * Get all viewers in the system.
     * 
     * @return An unmodifiable view of all viewers.
     */
    public Collection<ViewerProxy> getAllViewers() {
        synchronized (this) {
            return Collections.unmodifiableCollection(viewers);
        }
    }

    /**
     * Add a KernelListener.
     * 
     * @param l The listener to add.
     */
    public void addKernelListener(KernelListener l) {
        synchronized (listeners) {
            listeners.add(l);
        }
    }

    /**
     * Remove a KernelListener.
     * 
     * @param l The listener to remove.
     */
    public void removeKernelListener(KernelListener l) {
        synchronized (listeners) {
            listeners.remove(l);
        }
    }

    /**
     * Get the current time.
     * 
     * @return The current time.
     */
    public int getTime() {
        synchronized (this) {
            return time;
        }
    }

    /**
     * Get the world model.
     * 
     * @return The world model.
     */
    public WorldModel<? extends Entity> getWorldModel() {
        return worldModel;
    }

    /**
     * Find out if the kernel has terminated.
     * 
     * @return True if the kernel has terminated, false otherwise.
     */
    public boolean hasTerminated() {
        synchronized (this) {
            return isShutdown || termination.shouldStop(getState());
        }
    }

    /**
     * Run a single timestep.
     * 
     * @throws InterruptedException If this thread is interrupted during the
     *                              timestep.
     * @throws KernelException      If there is a problem executing the timestep.
     * @throws LogException         If there is a problem writing the log.
     */
    public void timestep() throws InterruptedException, KernelException, LogException {
        try {
            Logger.pushLogContext(KERNEL_LOG_CONTEXT);
            synchronized (this) {
                if (time == 0) {
                    fireStarted();
                }
                if (isShutdown) {
                    return;
                }
                ++time;

                // Work out what the agents can see and hear (using the commands from the
                // previous timestep).
                // Wait for new commands
                // Send commands to simulators and wait for updates
                // Collate updates and broadcast to simulators
                // Send perception, commands and updates to viewers
                Timestep nextTimestep = new Timestep(time);
                Logger.info("Timestep " + time);
                Logger.debug("Sending agent updates");
                long start = System.currentTimeMillis();
                JsonArray convertedObject = new JsonArray();
                long perceptionTime = 0;
                Collection<Command> commands = new ArrayList<Command>();
                if ((getTime() % config.getIntValue("episode.length") > 0)) {

                    try {
                        convertedObject = recvAction();
                        // System.out.println("converted object ######################### : " +
                        // convertedObject);
                        perceptionTime = System.currentTimeMillis();
                        Logger.debug("Waiting for commands");
                        commands = waitForCommands(time, convertedObject);
                    } catch (Exception e) {
                        System.out.println("converted object ######################### : " + convertedObject);
                        e.printStackTrace();
                    }

                }

                // JsonArray receivedInfo = sendAgentUpdates(nextTimestep, previousTimestep ==
                // null ? new HashSet<Command>() : previousTimestep.getCommands(),
                // previousTimestep == null ? new ChangeSet() :
                // previousTimestep.getChangeSet());

                //////////////////////////////////
                // double agent_reward = 0;
                // nextTimestep.setCommands(commands);
                // for(Command c : commands){
                // Entity e = worldModel.getEntity(c.getAgentID());
                // if(e instanceof FireBrigade){
                // FireBrigade f = (FireBrigade) e;
                // if(c instanceof AKExtinguish){
                // AKExtinguish a = (AKExtinguish) c;
                // Building b = (Building) worldModel.getEntity(a.getTarget());
                // if(b.isOnFire() && f.getWater() >= 3000){
                // agent_reward++;
                // }
                // }
                // else{
                // if(f.getWater() < 3000 && worldModel.getEntity(f.getPosition()) instanceof
                ////////////////////////////////// Refuge){
                // agent_reward=agent_reward + 0.1;
                // }
                // }
                // }
                // }
                // scoreMap.put("agentreward", 0.0);
                ///////////////////////////////////////

                // nextTimestep.setCommands(commands);
                // log.writeRecord(new CommandsRecord(time, commands));
                long commandsTime = System.currentTimeMillis();
                Logger.debug("Broadcasting commands");
                ResultfromSims results = sendCommandsToSimulators(time, commands);
                RewardSet rewards = results.rewardSet;
                ChangeSet changes = results.changeSet;
                // System.out.println(rewards);
                // simulatorUpdates.show(changes);
                System.out.println("the number of burning bulidings :: kerenl.java line 509 : "
                        + kernelFireBrigadeAgents.get(0).getBurningBuildings().size());

                if ((getTime() % config.getIntValue("episode.length") == 0)) {
                    System.out.println("Kernel : " + getTime());
                    try {
                        recvAction();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    ArrayList<Double> scoreDoubles = this.buildingDamageScore.score_abs_rel(worldModel, nextTimestep);
                    // ChangeSet changes2 = new ChangeSet();
                    // changes2.addAll(initialWorld.getAllEntities());
                    // for(Entity t : initialWorld.getAllEntities()){
                    // Entity target = worldModel.getEntity(t.getID());
                    // changes2.entityDeleted(t.getID());

                    // System.out.println("t : " + t.getClass().getSimpleName() +" " +
                    // t.getProperties().size());
                    // System.out.println("target : " + target.getClass().getSimpleName()+ " " +
                    // target.getProperties().size() );
                    // if(target != null)
                    // for(Property p : t.getProperties()){
                    // changes2.addChange(target, p);
                    // }

                    // }
                    String buildingdamage_str = gson.toJson(scoreDoubles);
                    changes = initialWorld;
                    output.println(buildingdamage_str + "#" + "@");

                    // System.out.println("the number of burning bulidings :: kerenl.java line 531
                    // :" + kernelFireBrigadeAgents.get(0).getBurningBuildings().size());

                }

                // nextTimestep.setChangeSet(changes);
                // log.writeRecord(new UpdatesRecord(time, changes));
                long updatesTime = System.currentTimeMillis();
                // Merge updates into world model

                worldModel.merge(changes);
                long mergeTime = System.currentTimeMillis();
                Logger.debug("Broadcasting updates");
                // System.out.println("the number of burning bulidings :: kerenl.java line 543 :
                // " + kernelFireBrigadeAgents.get(0).getBurningBuildings().size());
                sendUpdatesToSimulators(time, changes);
                // System.out.println("the number of burning bulidings :: kerenl.java line 546 :
                // " + kernelFireBrigadeAgents.get(0).getBurningBuildings().size());

                sendToViewers(nextTimestep);
                long broadcastTime = System.currentTimeMillis();
                Logger.debug("Computing score");
                // double s = score.score(worldModel, nextTimestep);
                long scoreTime = System.currentTimeMillis();
                // nextTimestep.setScore(s);
                // ScoreGraph sg = (ScoreGraph)score;
                // ScoreTable st = (ScoreTable)sg.child;
                // for(ScoreTable.ScoreModel.ScoreFunctionEntry entry : st.model.entries){
                // System.out.println(entry.getScoreFunctionName());
                // System.out.println(entry.getScore(time));

                // }

                HashMap<Integer, Integer> dispatchedTeamCounts = new HashMap<Integer, Integer>();
                if ((getTime() % config.getIntValue("episode.length") > 0)) {

                    String cs_str = gson.toJson(changes);
                    String rm_str = gson.toJson(rewards);
                    System.out.println("sm_str :" + rm_str);
                    for (Command c : commands) {
                        if ((c instanceof AKExtinguish)) {
                            AKExtinguish a = (AKExtinguish) c;
                            int eID = a.getTarget().getValue();
                            if (dispatchedTeamCounts.putIfAbsent(eID, 1) != null) {
                                dispatchedTeamCounts.put(eID, dispatchedTeamCounts.get(eID) + 1);
                            }
                        }
                    }
                    String dispatch_str = gson.toJson(dispatchedTeamCounts);
                    output.println(cs_str + "@" + rm_str + "@" + dispatch_str + "$");
                    // System.out.println(cs_str + "@" + rm_str + "@" + dispatch_str + "$");
                    // output.println(cs_str+ "@" + sm_str + "$");
                }
                Logger.info("Timestep " + time + " complete");
                // Logger.debug("Score: " + s);
                Logger.debug("Perception took        : " + (perceptionTime - start) + "ms");
                Logger.debug("Agent commands took    : " + (commandsTime - perceptionTime) + "ms");
                Logger.debug("Simulator updates took : " + (updatesTime - commandsTime) + "ms");
                Logger.debug("World model merge took : " + (mergeTime - updatesTime) + "ms");
                Logger.debug("Update broadcast took  : " + (broadcastTime - mergeTime) + "ms");
                Logger.debug("Score calculation took : " + (scoreTime - broadcastTime) + "ms");
                Logger.debug("Total time             : " + (scoreTime - start) + "ms");
                fireTimestepCompleted(nextTimestep);
                previousTimestep = nextTimestep;
                // Logger.debug("Commands: " + commands);
                // Logger.debug("Timestep commands: " + previousTimestep.getCommands());

            }
        } finally {
            Logger.popLogContext();
        }
    }

    /**
     * Shut down the kernel. This method will notify all agents/simulators/viewers
     * of the shutdown.
     */
    public void shutdown() {
        synchronized (this) {
            if (isShutdown) {
                return;
            }
            Logger.info("Kernel is shutting down");
            ExecutorService service = Executors.newFixedThreadPool(agents.size() + sims.size() + viewers.size());
            List<Callable<Object>> callables = new ArrayList<Callable<Object>>();
            for (AgentProxy next : agents) {
                final AgentProxy proxy = next;
                callables.add(Executors.callable(new Runnable() {
                    @Override
                    public void run() {
                        proxy.shutdown();
                    }
                }));
            }
            for (SimulatorProxy next : sims) {
                final SimulatorProxy proxy = next;
                callables.add(Executors.callable(new Runnable() {
                    @Override
                    public void run() {
                        proxy.shutdown();
                    }
                }));
            }
            for (ViewerProxy next : viewers) {
                final ViewerProxy proxy = next;
                callables.add(Executors.callable(new Runnable() {
                    @Override
                    public void run() {
                        proxy.shutdown();
                    }
                }));
            }
            try {
                service.invokeAll(callables);
            } catch (InterruptedException e) {
                Logger.warn("Interrupted during shutdown");
            }
            try {
                log.writeRecord(new EndLogRecord());
                log.close();
            } catch (LogException e) {
                Logger.error("Error closing log", e);
            }
            Logger.info("Kernel has shut down");
            isShutdown = true;
            fireShutdown();
        }
    }

    private JsonArray sendAgentUpdates(Timestep timestep, Collection<Command> commandsLastTimestep, ChangeSet cs)
            throws InterruptedException, KernelException, LogException {
        perception.setTime(time);
        communicationModel.process(time, commandsLastTimestep);
        // String cs_str = gson.toJson(cs);
        // String sm_str = gson.toJson(scoreMap);
        // String commandLastTimestep_str = gson.toJson(commandsLastTimestep);
        // System.out.println("ChangeSet");
        // System.out.println(cs_str);

        JsonArray convertedObject = new JsonArray();

        return convertedObject;
    }

    private JsonArray recvAction() throws Exception {
        System.out.println("input.readLine");
        byte[] data = new byte[4];
        // ????????? ????????? ?????????.
        input.read(data, 0, 4);
        System.out.println("input.readLine2");

        // ByteBuffer??? ?????? little ????????? ???????????? ????????? ????????? ?????????.
        ByteBuffer b = ByteBuffer.wrap(data);
        b.order(ByteOrder.LITTLE_ENDIAN);
        int length = b.getInt();
        // ???????????? ?????? ????????? ????????????.
        data = new byte[length];
        // ???????????? ?????????.
        input.read(data, 0, length);
        System.out.println("input.readLine3");

        String response = new String(data, "UTF-8");
        System.out.println("input.readLine4");

        JsonArray convertedObject = new Gson().fromJson(response, JsonArray.class);
        // System.out.println("input.readLine5" + convertedObject);

        return convertedObject;
    }

    private Collection<Command> waitForCommands(int timestep, JsonArray actions) throws InterruptedException {
        long beforeTime = System.currentTimeMillis();

        // Collection<Command> commands = commandCollector.getAgentCommands(agents,
        // timestep);
        Collection<Command> commands = new ArrayList<Command>();
        System.out.println(actions);
        int agentNum = 18;
        // class FBRunnable implements Runnable {
        // private int ts;
        // private JsonArray actions;
        // private Collection<Command> commands;
        // private int idx;
        //
        // public FBRunnable(int timestep, JsonArray actions, Collection<Command>
        // commands, int idx) {
        // this.ts = timestep;
        // this.actions = actions;
        // this.commands = commands;
        // this.idx = idx;
        // }
        //
        // public void run() {
        // int id = actions.get(this.idx).getAsInt();
        // KernelFireBrigadeAgent kernelFB = kernelFireBrigadeAgents.get(this.idx);
        // Command c = kernelFB.think(timestep, new EntityID(id), "cmd");
        // commands.add(c);
        // }
        // }
        // Thread[] threads = new Thread[agentNum];
        // for (int i = 0; i < agentNum; i++) {
        // FBRunnable fr = new FBRunnable(timestep, actions, commands, i);
        // threads[i] = new Thread(fr);
        // }
        // for (int i = 0; i < agentNum; i++) {
        // threads[i].start();
        // }
        // for (int i = 0; i < agentNum; i++) {
        // threads[i].join();
        // }

        int i = 0;
        for (KernelFireBrigadeAgent kernelFB : kernelFireBrigadeAgents) {
            int id = actions.get(i).getAsInt();

            Command c = kernelFB.think(timestep, new EntityID(id), "cmd");
            commands.add(c);
            i++;
        }

        // Logger.debug("Raw commands: " + commands);
        // long afterTime = System.currentTimeMillis();
        commandFilter.filter(commands, getState());
        // Logger.debug("Filtered commands: " + commands);
        // afterTime = System.currentTimeMillis();
        return commands;
    }

    /**
     * Send commands to all simulators and return which entities have been updated
     * by the simulators.
     */
    private ResultfromSims sendCommandsToSimulators(int timestep, Collection<Command> commands)
            throws InterruptedException {
        for (SimulatorProxy next : sims) {
            next.sendAgentCommands(timestep, commands);
        }
        // Wait until all simulators have sent updates
        ChangeSet result = new ChangeSet();
        RewardSet result_reward = new RewardSet();
        for (SimulatorProxy next : sims) {
            Logger.debug("Fetching updates from " + next);
            result.merge(next.getUpdates(timestep));
            result_reward.merge(next.getUpdatesReward(timestep));

        }
        return new ResultfromSims(result, result_reward);
    }

    private void sendUpdatesToSimulators(int timestep, ChangeSet updates) throws InterruptedException {
        for (SimulatorProxy next : sims) {
            next.sendUpdate(timestep, updates);
        }
    }

    private void sendToViewers(Timestep timestep) {
        for (ViewerProxy next : viewers) {
            next.sendTimestep(timestep);
        }
    }

    private Set<KernelListener> getListeners() {
        Set<KernelListener> result;
        synchronized (listeners) {
            result = new HashSet<KernelListener>(listeners);
        }
        return result;
    }

    private void fireStarted() {
        for (KernelListener next : getListeners()) {
            next.simulationStarted(this);
        }
    }

    private void fireShutdown() {
        for (KernelListener next : getListeners()) {
            next.simulationEnded(this);
        }
    }

    private void fireTimestepCompleted(Timestep timestep) {
        for (KernelListener next : getListeners()) {
            next.timestepCompleted(this, timestep);
        }
    }

    private void fireAgentAdded(AgentProxy agent) {
        for (KernelListener next : getListeners()) {
            next.agentAdded(this, agent);
        }
    }

    private void fireAgentRemoved(AgentProxy agent) {
        for (KernelListener next : getListeners()) {
            next.agentRemoved(this, agent);
        }
    }

    private void fireSimulatorAdded(SimulatorProxy sim) {
        for (KernelListener next : getListeners()) {
            next.simulatorAdded(this, sim);
        }
    }

    private void fireSimulatorRemoved(SimulatorProxy sim) {
        for (KernelListener next : getListeners()) {
            next.simulatorRemoved(this, sim);
        }
    }

    private void fireViewerAdded(ViewerProxy viewer) {
        for (KernelListener next : getListeners()) {
            next.viewerAdded(this, viewer);
        }
    }

    private void fireViewerRemoved(ViewerProxy viewer) {
        for (KernelListener next : getListeners()) {
            next.viewerRemoved(this, viewer);
        }
    }
}
