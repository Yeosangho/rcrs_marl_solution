package rescuecore2.worldmodel;

import static rescuecore2.misc.EncodingTools.writeInt32;
import static rescuecore2.misc.EncodingTools.writeFloat32;
import static rescuecore2.misc.EncodingTools.writeProperty;
import static rescuecore2.misc.EncodingTools.readInt32;
import static rescuecore2.misc.EncodingTools.readFloat32;
import static rescuecore2.misc.EncodingTools.readProperty;

import java.util.Set;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.List;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import rescuecore2.misc.collections.LazyMap;

/**
   Abstract base class for concrete Entity implementations.
 */
public class RewardSet  {
  private Map<Integer, Float> rewards;
  
  public RewardSet(){
    rewards = new LazyMap<Integer, Float>(){
        @Override
        public Float createValue(){
            return (float)0.0;
        }
    };
  }
  public RewardSet(RewardSet other){
      this();
      merge(other);
  }

  public void addReward(Integer id, Float reward){
    rewards.put(id, reward);
  }


  public void merge(RewardSet other){
      for(Map.Entry<Integer, Float> next : other.rewards.entrySet()){
          Integer e = next.getKey();
          addReward(e, next.getValue());

      }
  }

  public void write(OutputStream out) throws IOException{
      writeInt32(rewards.size(), out);
      for ( Map.Entry<Integer, Float> next : rewards.entrySet() ) {  
          Integer id = next.getKey();
          Float values = next.getValue();
          writeInt32(id, out);
          writeFloat32(values, out);
      }
  }
  public String toString(){
    String result = "";
    for(Map.Entry<Integer, Float> next : rewards.entrySet()){
        Integer e = next.getKey();
        Float rewardList = next.getValue();
        result += Integer.toString(e) + ":" + rewardList.toString() + "\n";
    }
    return result;
  }
  public void read(InputStream in) throws IOException {
      int entityCount = readInt32(in);
      for(int i=0; i<entityCount; ++i){
          Integer id = readInt32(in);
          Float values = readFloat32(in);
          addReward(id, values);
          
      }
  }

  public Set<Integer> getChangedEntities() {
    return new HashSet<Integer>( rewards.keySet() );
  }
}
