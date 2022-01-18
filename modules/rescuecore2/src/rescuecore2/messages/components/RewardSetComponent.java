package rescuecore2.messages.components;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;

import rescuecore2.worldmodel.RewardSet;
import rescuecore2.messages.AbstractMessageComponent;

public class RewardSetComponent extends AbstractMessageComponent {
    private RewardSet rewards;

    public RewardSetComponent(String name){
        super(name);
        rewards = new RewardSet();
    }

    public RewardSetComponent(String name, RewardSet rewards){
        super(name);
        this.rewards = new RewardSet(rewards);
    }
    public RewardSet getRewardSet(){
        return rewards;
    }

    public void setRewardSet(RewardSet newRewards){
        this.rewards = new RewardSet(newRewards);
    }

    @Override
    public void write(OutputStream out) throws IOException {
        rewards.write(out);
    }

    @Override
    public void read(InputStream in) throws IOException {
        rewards = new RewardSet();
        rewards.read(in);
    }

    @Override
    public String toString() {
        return getName() + " = " + rewards.getChangedEntities().size() + " entities";
    }
}