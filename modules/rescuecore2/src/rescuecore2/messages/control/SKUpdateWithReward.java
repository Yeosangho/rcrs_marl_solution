package rescuecore2.messages.control;

import java.io.InputStream;
import java.io.IOException;

import rescuecore2.messages.components.RewardSetComponent;
import rescuecore2.worldmodel.ChangeSet;
import rescuecore2.worldmodel.RewardSet;


public class SKUpdateWithReward extends SKUpdate{
    private RewardSetComponent rewardupdate;

    public SKUpdateWithReward(InputStream in) throws IOException{
        this();
        read(in);
    }
    public SKUpdateWithReward(int id, int time, ChangeSet changes, RewardSet rewards){
        this();
        this.id.setValue(id);
        this.time.setValue(time);
        this.update.setChangeSet(changes);        
        this.rewardupdate.setRewardSet(rewards);
    }
    
    private SKUpdateWithReward(){
        super(ControlMessageURN.SK_UPDATE_REWARD);
        rewardupdate = new RewardSetComponent("Rewards");
        addMessageComponent(rewardupdate);
    }

    public RewardSet getRewardSet(){
        return rewardupdate.getRewardSet();
    }
}