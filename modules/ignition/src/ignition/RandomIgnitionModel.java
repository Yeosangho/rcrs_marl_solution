package ignition;

import rescuecore2.config.Config;
import rescuecore2.log.Logger;
import java.util.Random;

import rescuecore2.standard.entities.StandardWorldModel;
import rescuecore2.standard.entities.StandardEntity;
import rescuecore2.standard.entities.Building;
import rescuecore2.Constants;

import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Collections;

import org.uncommons.maths.random.PoissonGenerator;

/**
   An IgnitionModel that ignites unburnt buildings in a random order. The number of ignitions per timestep is drawn from a Poisson distribution.
*/
public class RandomIgnitionModel implements IgnitionModel {
    private static final String MEAN_KEY = "ignition.random.lambda";

    private PoissonGenerator generator;
    private Iterator<Building> it;
    private Random shuffleRandom;
    private Random poissonRandom;
    private List<Building> unburnt;
    /**
       Construct a RandomIgnitionModel.
       @param world The world model.
       @param config The system configuration.
    */
    public RandomIgnitionModel(StandardWorldModel world, Config config) {
        String seed= config.getValue(Constants.RANDOM_SEED_KEY, "");
        unburnt = new ArrayList<Building>();
        for (StandardEntity next : world) {
            if (next instanceof Building) {
                unburnt.add((Building)next);
            }
        }
        shuffleRandom = new Random(Integer.parseInt(seed));
        Collections.shuffle(unburnt, shuffleRandom);
        double mean = config.getFloatValue(MEAN_KEY);
        poissonRandom = new Random(Integer.parseInt(seed));
        generator = new PoissonGenerator(mean, poissonRandom);
        it = unburnt.iterator();
    }
    public void reset(StandardWorldModel world, Config config){
        String seed= config.getValue(Constants.RANDOM_SEED_KEY, "");
        unburnt = new ArrayList<Building>();
        for (StandardEntity next : world) {
            if (next instanceof Building) {
                unburnt.add((Building)next);
            }
        }
        shuffleRandom = new Random(Integer.parseInt(seed));
        Collections.shuffle(unburnt, shuffleRandom);
        double mean = config.getFloatValue(MEAN_KEY);
        poissonRandom = new Random(Integer.parseInt(seed));
        generator = new PoissonGenerator(mean, poissonRandom);
        it = unburnt.iterator();
    }
    @Override
    public Set<Building> findIgnitionPoints(StandardWorldModel world, int time) {
        Set<Building> result = new HashSet<Building>();
        if (it.hasNext()) {
            int number = generator.nextValue();
            //number = 1;
            Logger.debug("Igniting " + number + " buildings");
            for (int i = 0; i < number && it.hasNext(); ++i) {
                result.add(it.next());
            }
        }
        return result;
    }
}
