package ignition;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import rescuecore2.log.Logger;
import rescuecore2.messages.control.KSCommands;
import rescuecore2.standard.components.StandardSimulator;
import rescuecore2.standard.entities.Building;
import rescuecore2.standard.entities.GasStation;
import rescuecore2.standard.entities.StandardEntity;
import rescuecore2.standard.entities.StandardEntityURN;
import rescuecore2.worldmodel.ChangeSet;
import rescuecore2.worldmodel.RewardSet;




/**
   A simulator that determines when new building fires begin.
*/
public class IgnitionSimulator extends StandardSimulator {
    private RandomIgnitionModel ignitionModel;
    private int GAS_STATION_EXPLOSION_RANG;
    private List<GasStation> notIgnaitedGasStations;
    private List<Building> burntBuildings;
    @Override
    protected void postConnect() {
        super.postConnect();
        burntBuildings = new ArrayList<Building>();
        ignitionModel = new RandomIgnitionModel(model, config);
        GAS_STATION_EXPLOSION_RANG=config.getIntValue("ignition.gas_station.explosion.range");
        notIgnaitedGasStations=new ArrayList<GasStation>();
        for (StandardEntity entity : model.getEntitiesOfType(StandardEntityURN.GAS_STATION)) {
			notIgnaitedGasStations.add((GasStation) entity);
		}
    }
    @Override
    protected void processCommands(KSCommands c, ChangeSet changes, RewardSet rewards) {
        long start = System.currentTimeMillis();
        int time = c.getTime();
        Logger.info("Timestep " + time);
		if((c.getTime()) % config.getIntValue("episode.length") >= config.getIntValue("episode.length")-1){
            super.postConnect();
            burntBuildings = new ArrayList<Building>();
            ignitionModel.reset(model, config);
            GAS_STATION_EXPLOSION_RANG=config.getIntValue("ignition.gas_station.explosion.range");
            notIgnaitedGasStations=new ArrayList<GasStation>();
            for (StandardEntity entity : model.getEntitiesOfType(StandardEntityURN.GAS_STATION)) {
		    	notIgnaitedGasStations.add((GasStation) entity);
		    }
            for(Building next : burntBuildings){
                next.setIgnition(false);
                changes.addChange(next, next.getIgnitionProperty());
            }
		}
        else{
            explosionGasStations(changes);
            
            Logger.info("Ignating after shock ");
            // Find out which buildings have ignited.
            Set<Building> buildings = ignitionModel.findIgnitionPoints(model, c.getTime()%config.getIntValue("episode.length"));
            for (Building next : buildings) {
                Logger.info("Igniting " + next);
                next.setIgnition(true);
                changes.addChange(next, next.getIgnitionProperty());
                burntBuildings.add(next);
            }     
        }  
        long end = System.currentTimeMillis();
        Logger.info("Timestep " + time + " took " + (end - start) + " ms");    
    }
    @Override
    protected void processCommands(KSCommands c, ChangeSet changes) {
        long start = System.currentTimeMillis();
        int time = c.getTime();
        Logger.info("Timestep " + time);

        explosionGasStations(changes);

        Logger.info("Ignating after shock ");
        // Find out which buildings have ignited.
        Set<Building> buildings = ignitionModel.findIgnitionPoints(model, c.getTime());
        for (Building next : buildings) {
            Logger.info("Igniting " + next);
            next.setIgnition(true);
            changes.addChange(next, next.getIgnitionProperty());
        }
        long end = System.currentTimeMillis();
        Logger.info("Timestep " + time + " took " + (end - start) + " ms");
    }

    private void explosionGasStations(ChangeSet changes) {
    	Logger.info("explosion Gas Stations ");
    	for (Iterator<GasStation> iterator= notIgnaitedGasStations.iterator(); iterator.hasNext();) {
    		GasStation gasStation = iterator.next();
        	if(gasStation.isFierynessDefined()&&gasStation.getFieryness()==1){
        		explode(gasStation,changes);
        		iterator.remove();
        	}
        }
	}

	private void explode(GasStation gasStation, ChangeSet changes) {
		Logger.info(gasStation+" Ignited ==> explosion" );
		for (StandardEntity rangeEntity : model.getObjectsInRange(gasStation, GAS_STATION_EXPLOSION_RANG)) {
			if(rangeEntity instanceof Building){
				Building rangeBuilding = (Building)rangeEntity;
				Logger.info("Igniting " + rangeBuilding);
				rangeBuilding.setIgnition(true);
	            changes.addChange(rangeBuilding, rangeBuilding.getIgnitionProperty());
			}
		}
	}

	@Override
    public String getName() {
        return "Ignition simulator";
    }
}