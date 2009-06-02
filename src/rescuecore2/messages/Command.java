package rescuecore2.messages;

import rescuecore2.worldmodel.EntityID;

/**
   A sub-interface of Message that tags messages that are interpreted as agent commands.
 */
public interface Command {
    /**
       Get the id of the agent-controlled entity that has issued this command.
       @return The id of the agent.
     */
    EntityID getAgentID();
}