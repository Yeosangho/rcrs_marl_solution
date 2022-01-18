package rescuecore2.messages.control;

import rescuecore2.messages.Control;
import rescuecore2.messages.Command;
import rescuecore2.messages.AbstractMessage;
import rescuecore2.messages.components.IntComponent;
import rescuecore2.messages.components.EntityIDComponent;
import rescuecore2.messages.components.ChangeSetComponent;
import rescuecore2.messages.components.CommandListComponent;
import rescuecore2.messages.components.StringComponent;
import rescuecore2.worldmodel.EntityID;
import rescuecore2.worldmodel.ChangeSet;

import java.io.InputStream;
import java.io.IOException;
import java.util.Collection;

/**
   A message for signalling a perception update for an agent.
 */
public class KACommand extends KASense {
    private EntityIDComponent buildingID;
    private StringComponent command;

    /**
       A KASense message that populates its data from a stream.
       @param in The InputStream to read.
       @throws IOException If there is a problem reading the stream.
     */
    public KACommand(InputStream in) throws IOException {
        this();
        read(in);
    }

    /**
       A populated KASense message.
       @param agentID The ID of the Entity that is receiving the update.
       @param time The timestep of the simulation.
       @param changes All changes that the agent can perceive.
       @param hear The messages that the agent can hear.
     */
    public KACommand(EntityID agentID, int time, ChangeSet changes, Collection<? extends Command> hear, EntityID bID, String command) {
     
        this();
        this.agentID.setValue(agentID);
        this.time.setValue(time);
        this.updates.setChangeSet(changes);
        this.hear.setCommands(hear);
        this.buildingID.setValue(bID);
        this.command.setValue(command);
    }

    private KACommand() {
        super(ControlMessageURN.KA_COMMAND);
        buildingID = new EntityIDComponent("Agent ID");
        command = new StringComponent("Command");

        addMessageComponent(buildingID);
        addMessageComponent(command);
    }

    public EntityID getBuildingID(){
        return buildingID.getValue();
    }

    public String getCommand(){
        return command.getValue();
    }

}
