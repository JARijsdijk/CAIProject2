import sys, random, enum, ast, time, csv
import numpy as np
from matrx import grid_world
from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import *
from matrx import utils
from matrx.grid_world import GridWorld
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.move_actions import MoveNorth
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
from actions1.CustomActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop


class Phase(enum.Enum):
    INTRO = 1,
    FIND_NEXT_GOAL = 2,
    PICK_UNSEARCHED_ROOM = 3,
    PLAN_PATH_TO_ROOM = 4,
    FOLLOW_PATH_TO_ROOM = 5,
    PLAN_ROOM_SEARCH_PATH = 6,
    FOLLOW_ROOM_SEARCH_PATH = 7,
    PLAN_PATH_TO_VICTIM = 8,
    FOLLOW_PATH_TO_VICTIM = 9,
    TAKE_VICTIM = 10,
    PLAN_PATH_TO_DROPZONE = 11,
    FOLLOW_PATH_TO_DROPZONE = 12,
    DROP_VICTIM = 13,
    WAIT_FOR_HUMAN = 14,
    WAIT_AT_DROPZONE = 15,
    FIX_ORDER_GRAB = 16,
    FIX_ORDER_DROP = 17,
    REMOVE_OBSTACLE_IF_NEEDED = 18,
    ENTER_ROOM = 19


def _get_drop_zones(state):
    """
    Returns the list of drop zones (their full dict),
    in order (the first one is the place that requires the first drop)
    """
    places = state[{'is_goal_block': True}]
    places.sort(key=lambda info: info['location'][1])
    zones = []
    for place in places:
        if place['drop_zone_nr'] == 0:
            zones.append(place)
    return zones


def _efficient_search(tiles):
    """
    Efficiently transverse areas instead of moving over every single area tile
    """
    x = []
    y = []
    for i in tiles:
        if i[0] not in x:
            x.append(i[0])
        if i[1] not in y:
            y.append(i[1])
    locations = []
    for i in range(len(x)):
        if i % 2 == 0:
            locations.append((x[i], min(y)))
        else:
            locations.append((x[i], max(y)))
    return locations


class BaselineAgent(ArtificialBrain):
    def __init__(self, slowdown, condition, name, folder):
        super().__init__(slowdown, condition, name, folder)
        # Initialization of some relevant variables
        self._slowdown = slowdown
        self._condition = condition
        self._humanName = name
        self._folder = folder
        self._phase = Phase.INTRO
        self._door = None
        self._doormat = None
        self._navigator = None
        self._roomTiles = None
        self._state_tracker = None
        self._victimsInRoom = []
        self._searchedRooms = []
        self._foundVictims = []
        self._collectedVictims = []
        self._foundVictimLocations = {}
        self._sendMessages = []
        self._currentDoor = None
        self._teamMembers = []
        self._carryingTogether = False
        self._remove = False
        self._targetVictim = None
        self._targetDropZone = None
        self._humanLocation = None
        self._distanceToHuman = None
        self._distanceToDropZone = None
        self._agentLocation = None
        self._todo = []
        self._answered = False
        self._toSearch = []
        self._carrying = False
        self._waiting = False
        self._rescue = None
        self._recentVictim = None
        self._moving = False
        self._remainingVictims = []
        self._remainingDropZones = {}

        # Additional variables used for our trust implementation
        # Create a dictionary with trust values for all team members
        self._trustBeliefs = {}

    def initialize(self):
        # Initialization of the state tracker and navigation algorithm
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state):
        # Filtering of the world state before deciding on an action 
        return state

    def decide_on_actions(self, state):
        # Identify team members
        agent_name = state[self.agent_id]['obj_id']
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        # Initialize and update trust beliefs for team members
        self._load_beliefs()
        # Process messages from team members
        self._process_messages(state)

        # Check whether human is close in distance
        if state[{'is_human_agent': True}]:
            self._distanceToHuman = 'close'
        if not state[{'is_human_agent': True}]:
            # Define distance between human and agent based on last known area locations
            if self._agentLocation in [1, 2, 3, 4, 5, 6, 7] and self._humanLocation in [8, 9, 10, 11, 12, 13, 14]:
                self._distanceToHuman = 'far'
            if self._agentLocation in [1, 2, 3, 4, 5, 6, 7] and self._humanLocation in [1, 2, 3, 4, 5, 6, 7]:
                self._distanceToHuman = 'close'
            if self._agentLocation in [8, 9, 10, 11, 12, 13, 14] and self._humanLocation in [1, 2, 3, 4, 5, 6, 7]:
                self._distanceToHuman = 'far'
            if self._agentLocation in [8, 9, 10, 11, 12, 13, 14] and self._humanLocation in [8, 9, 10, 11, 12, 13, 14]:
                self._distanceToHuman = 'close'

        # Define distance to drop zone based on last known area location
        if self._agentLocation in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distanceToDropZone = 'far'
        if self._agentLocation in [3, 4, 7, 10, 13, 14]:
            self._distanceToDropZone = 'close'

        # Check whether a victim is currently being carried together by the human and agent
        # TODO stop willingness decay when carrying together?
        for info in state.values():
            if 'is_human_agent' in info and self._humanName in info['name'] and len(info['is_carrying']) > 0:
                if 'critical' in info['is_carrying'][0]['obj_id'] \
                 or 'mild' in info['is_carrying'][0]['obj_id'] and self._rescue == 'together' and not self._moving:
                    # If a victim is being carried,
                    # then add them to the list of collected victims
                    # (if they have not already been added)
                    if info['is_carrying'][0]['img_name'][8:-4] not in self._collectedVictims:
                        self._collectedVictims.append(info['is_carrying'][0]['img_name'][8:-4])
                        self._carryingTogether = True
            if 'is_human_agent' in info and self._humanName in info['name'] and len(info['is_carrying']) == 0:
                self._carryingTogether = False
        # If carrying a victim together,
        # then let agent be idle (because joint actions are essentially carried out by the human)
        if self._carryingTogether:
            return None, {}

        # Send the hidden score message for displaying and logging the score during the task, DO NOT REMOVE THIS
        self._send_message('Our score is ' + str(state['rescuebot']['score']) + '.', 'RescueBot')

        # Ongoing loop until the task is terminated, using different phases for defining the agent's behavior
        while True:
            if Phase.INTRO == self._phase:
                # Send introduction message
                self._send_message('Hello! My name is RescueBot. \
                Together we will collaborate and try to search and rescue the 8 victims on our right as quickly as possible. \
                Each critical victim (critically injured girl/critically injured elderly woman/critically injured man/critically injured dog) adds 6 points to our score, \
                each mild victim (mildly injured boy/mildly injured elderly man/mildly injured woman/mildly injured cat) 3 points. \
                If you are ready to begin our mission, you can simply start moving.', 'RescueBot')
                # Wait until the human starts moving before going to the next phase, otherwise remain idle
                if not state[{'is_human_agent': True}]:
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}

            if Phase.FIND_NEXT_GOAL == self._phase:
                # Definition of some relevant variables
                self._answered = False
                self._targetVictim = None
                self._targetDropZone = None
                self._rescue = None
                self._moving = True
                # Identify which victims still need to be rescued and where they should be dropped-off
                drop_zones = _get_drop_zones(state)
                remaining_drop_zones = {}
                for info in drop_zones:
                    if str(info['img_name'])[8:-4] not in self._collectedVictims:
                        remaining_drop_zones[str(info['img_name'])[8:-4]] = info['location']
                    self._remainingDropZones = remaining_drop_zones
                self._remainingVictims = remaining_drop_zones.keys()
                # Remain idle if there are no victims left to rescue
                if len(self._remainingVictims) == 0:
                    return None, {}

                # Check which of the remaining victims can be rescued because the human or agent has already found them
                for victim in self._remainingVictims:
                    # Define a previously found victim as target victim because all areas have been searched
                    if victim in self._foundVictims and victim in self._todo and len(self._searchedRooms) == 0:
                        self._targetVictim = victim
                        self._targetDropZone = remaining_drop_zones[victim]
                        # TODO decide if we should request for help
                        self._rescue = 'together'
                        self._send_message('Moving to ' + self._foundVictimLocations[victim]['room']
                                           + ' to pick up ' + self._targetVictim
                                           + '. Please come there as well to help me carry '
                                           + self._targetVictim + ' to the drop zone.', 'RescueBot')
                        # Plan path
                        if 'location' in self._foundVictimLocations[victim].keys():
                            # Plan path to victim because the exact location is known
                            # (i.e., the agent found this victim)
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        else:
                            # Plan path to area because the exact victim location is not known, only the area
                            # (i.e., human found this victim)
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}

                    # Define a previously found victim as target victim
                    if victim in self._foundVictims and victim not in self._todo:
                        self._targetVictim = victim
                        self._targetDropZone = remaining_drop_zones[victim]
                        # TODO decide if we should request for help
                        # Rescue together when victim is critical,
                        # or when the human is weak and the victim is mildly injured
                        if 'critical' in victim or 'mild' in victim and self._condition == 'weak':
                            self._rescue = 'together'
                        # Rescue alone if the victim is mildly injured and the human not weak
                        if 'mild' in victim and self._condition != 'weak':
                            # DEBUG
                            print("INFO: Rescuing a mildly injured victim alone.")
                            self._rescue = 'alone'
                        # Plan path
                        if 'location' in self._foundVictimLocations[victim].keys():
                            # Plan path to victim because the exact location is known
                            # (i.e., the agent found this victim)
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        else:
                            # Plan path to area because the exact victim location is not known, only the area
                            # (i.e., human found this  victim)
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}

                    # If there are no remaining victims that have been found,
                    # then visit an unsearched area to search for victims
                    if victim not in self._foundVictims or victim in self._foundVictims \
                            and victim in self._todo and len(self._searchedRooms) > 0:
                        self._phase = Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM == self._phase:
                agent_location = state[self.agent_id]['location']
                # Identify which areas are not explored yet
                unsearched_rooms = [room['room_name'] for room in state.values()
                                    if 'class_inheritance' in room
                                    and 'Door' in room['class_inheritance']
                                    and room['room_name'] not in self._searchedRooms
                                    and room['room_name'] not in self._toSearch]

                # If all areas have been searched but the task is not finished,
                # then start searching areas again
                # TODO perhaps we should modify our trust in the human then
                if len(self._remainingDropZones) and len(unsearched_rooms) == 0:
                    self._toSearch = []
                    self._searchedRooms = []
                    self._sendMessages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._send_message('Going to re-search all areas.', 'RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL

                # If there are still areas to search, define which one to search next
                else:
                    # Identify the closest door when the agent did not search any areas yet
                    if self._currentDoor is None:
                        # Find all area entrance locations
                        self._door = state.get_room_doors(self._get_closest_room(state, unsearched_rooms, agent_location))[0]
                        self._doormat = state.get_room(self._get_closest_room(state, unsearched_rooms, agent_location))[-1]['doormat']
                        # Workaround for one area because of some bug
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        # Plan path to area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Identify the closest door when the agent just searched another area
                    if self._currentDoor is not None:
                        self._door = state.get_room_doors(self._get_closest_room(state, unsearched_rooms, self._currentDoor))[0]
                        self._doormat = state.get_room(self._get_closest_room(state, unsearched_rooms, self._currentDoor))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                # Switch to a different area when the human found a victim
                # TODO decide if we believe the human
                if self._targetVictim and self._targetVictim in self._foundVictims \
                        and 'location' not in self._foundVictimLocations[self._targetVictim].keys():
                    self._door = state.get_room_doors(self._foundVictimLocations[self._targetVictim]['room'])[0]
                    self._doormat = state.get_room(self._foundVictimLocations[self._targetVictim]['room'])[-1]['doormat']
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)
                    door_location = self._doormat
                # Otherwise plan the route to the previously identified area to search
                else:
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)
                    door_location = self._doormat
                self._navigator.add_waypoints([door_location])
                # Follow the route to the next area to search
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                # Find the next victim to rescue if the previously identified target victim was rescued by the human
                # TODO decide if we believe the human
                if self._targetVictim and self._targetVictim in self._collectedVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Identify which area to move to because the human found the previously identified target victim
                # TODO decide if we believe the human
                if self._targetVictim and self._targetVictim in self._foundVictims \
                        and self._door['room_name'] != self._foundVictimLocations[self._targetVictim]['room']:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Identify the next area to search if the human already searched the previously identified area
                # TODO decide if we believe the human
                if self._door['room_name'] in self._searchedRooms and self._targetVictim not in self._foundVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Otherwise move to the next area to search
                else:
                    self._state_tracker.update(state)
                    # Explain why the agent is moving to the specific area,
                    # either because it contains the current target victim or because it is the closest unsearched area
                    # TODO
                    if self._targetVictim in self._foundVictims \
                            and str(self._door['room_name']) == self._foundVictimLocations[self._targetVictim]['room'] \
                            and not self._remove:
                        if self._condition == 'weak':
                            self._send_message('Moving to ' + str(self._door['room_name']) + ' to pick up '
                                               + self._targetVictim + ' together with you.', 'RescueBot')
                        else:
                            self._send_message('Moving to ' + str(self._door['room_name']) + ' to pick up '
                                               + self._targetVictim + '.', 'RescueBot')
                    if self._targetVictim not in self._foundVictims and not self._remove or not self._targetVictim and not self._remove:
                        self._send_message('Moving to ' + str(self._door['room_name'])
                                           + ' because it is the closest unsearched area.', 'RescueBot')
                    self._currentDoor = self._door['location']
                    # Retrieve move actions to execute
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action is not None:
                        # Remove obstacles blocking the path to the area 
                        for info in state.values():
                            # TODO small brown stone in the way
                            if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] \
                                    and 'stone' in info['obj_id'] and info['location'] not in [(9, 4), (9, 7), (9, 19), (21, 19)]:
                                self._send_message('Reaching ' + str(self._door['room_name'])
                                                   + ' will take a bit longer because I found stones blocking my path.', 'RescueBot')
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                        return action, {}
                    # Identify and remove obstacles if they are blocking the entrance of the area
                    self._phase = Phase.REMOVE_OBSTACLE_IF_NEEDED

            if Phase.REMOVE_OBSTACLE_IF_NEEDED == self._phase:
                objects = []
                # Identify which obstacle is blocking the entrance
                for info in state.values():
                    # Case 1: Big gray rock
                    # TODO decide what to do about obstacle
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'rock' in info['obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if not self._answered and not self._remove and not self._waiting:
                            self._send_message('Found rock blocking ' + str(self._door['room_name']) +
                                               '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                                Important features to consider are: \n \
                                                safe - victims rescued: ' + str(self._collectedVictims) + ' \n \
                                                explore - areas searched: area ' + str(self._searchedRooms).replace('area ', '') + ' \n \
                                                clock - removal time: 5 seconds \n \
                                                afstand - distance between us: ' + self._distanceToHuman, 'RescueBot')
                            self._waiting = True                          
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._toSearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Wait for the human to help removing the obstacle and remove the obstacle together
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove' or self._remove:
                            if not self._remove:
                                self._answered = True
                            # Tell the human to come over and be idle until human arrives
                            # TODO wait for how long
                            if not state[{'is_human_agent': True}]:
                                self._send_message('Please come to ' + str(self._door['room_name']) + ' to remove rock.', 'RescueBot')
                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                self._send_message('Lets remove rock blocking ' + str(self._door['room_name']) + '!', 'RescueBot')
                                return None, {}
                        # Remain idle until the human communicates what to do with the identified obstacle
                        # TODO wait for how long
                        else:
                            return None, {}

                    # Case 2: Tree
                    # TODO decide what to do about obstacle
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info['obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if not self._answered and not self._remove and not self._waiting:
                            self._send_message('Found tree blocking  ' + str(self._door['room_name']) +
                                               '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                                Important features to consider are: \n \
                                                safe - victims rescued: ' + str(self._collectedVictims) + '\n \
                                                explore - areas searched: area ' + str(self._searchedRooms).replace('area ', '') + ' \n \
                                                clock - removal time: 10 seconds', 'RescueBot')
                            self._waiting = True
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._toSearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Remove the obstacle if the human tells the agent to do so
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove' or self._remove:
                            if not self._remove:
                                self._answered = True
                                self._waiting = False
                                self._send_message('Removing tree blocking ' + str(self._door['room_name']) + '.', 'RescueBot')
                            if self._remove:
                                self._send_message('Removing tree blocking ' + str(self._door['room_name']) + ' because you asked me to.', 'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remain idle until the human communicates what to do with the identified obstacle
                        # TODO wait for how long
                        else:
                            return None, {}

                    # Case 3: Small brown stone
                    # TODO decide what to do about obstacle
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if not self._answered and not self._remove and not self._waiting:
                            self._send_message('Found stones blocking  ' + str(self._door['room_name']) +
                                               '. Please decide whether to "Remove together", "Remove alone", or "Continue" searching. \n \n \
                                                Important features to consider are: \n \
                                                safe - victims rescued: ' + str(self._collectedVictims) + ' \n \
                                                explore - areas searched: area ' + str(self._searchedRooms).replace('area', '') + ' \n \
                                                clock - removal time together: 3 seconds \n \
                                                afstand - distance between us: ' + self._distanceToHuman + '\n \
                                                clock - removal time alone: 20 seconds', 'RescueBot')
                            self._waiting = True
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle          
                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._toSearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Remove the obstacle alone if the human decides so
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove alone' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            self._send_message('Removing stones blocking ' + str(self._door['room_name']) + '.', 'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remove the obstacle together if the human decides so
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove together' or self._remove:
                            if not self._remove:
                                self._answered = True
                            # Tell the human to come over and be idle until human arrives
                            # TODO wait for how long
                            if not state[{'is_human_agent': True}]:
                                self._send_message('Please come to ' + str(self._door['room_name']) + ' to remove stones together.', 'RescueBot')
                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                self._send_message('Lets remove stones blocking ' + str(self._door['room_name']) + '!', 'RescueBot')
                                return None, {}
                        # Remain idle until the human communicates what to do with the identified obstacle
                        # TODO wait for how long
                        else:
                            return None, {}

                # If no obstacles are blocking the entrance, enter the area
                if len(objects) == 0:
                    self._answered = False
                    self._remove = False
                    self._waiting = False
                    self._phase = Phase.ENTER_ROOM

            if Phase.ENTER_ROOM == self._phase:
                # TODO decide if we trust what the human says
                self._answered = False
                # If the target victim is rescued by the human, identify the next victim to rescue
                if self._targetVictim in self._collectedVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # If the target victim is found in a different area, start moving there
                if self._targetVictim in self._foundVictims and self._door['room_name'] != self._foundVictimLocations[self._targetVictim]['room']:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # If the human searched the same area, plan searching another area instead
                if self._door['room_name'] in self._searchedRooms and self._targetVictim not in self._foundVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Otherwise, enter the area and plan to search it
                else:
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action is not None:
                        return action, {}
                    self._phase = Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH == self._phase:
                self._agentLocation = int(self._door['room_name'].split()[-1])
                # Store the locations of all area tiles
                room_tiles = [info['location'] for info in state.values()
                              if 'class_inheritance' in info
                              and 'AreaTile' in info['class_inheritance']
                              and 'room_name' in info
                              and info['room_name'] == self._door['room_name']]
                self._roomTiles = room_tiles
                # Make the plan for searching the area
                self._navigator.reset_full()
                self._navigator.add_waypoints(_efficient_search(room_tiles))
                self._victimsInRoom = []
                self._phase = Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH == self._phase:
                # Search the area
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    # Identify victims present in the area
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            victim = str(info['img_name'][8:-4])
                            # Remember which victim the agent found in this area
                            if victim not in self._victimsInRoom:
                                self._victimsInRoom.append(victim)

                            # Identify the exact location of the victim that was found by the human earlier
                            if victim in self._foundVictims and 'location' not in self._foundVictimLocations[victim].keys():
                                self._recentVictim = victim
                                # Add the exact victim location to the corresponding dictionary
                                self._foundVictimLocations[victim] = {'location': info['location'], 'room': self._door['room_name'], 'obj_id': info['obj_id']}
                                if victim == self._targetVictim:
                                    # TODO modify trust
                                    # Communicate which victim was found
                                    self._send_message('Found ' + victim + ' in ' + self._door['room_name']
                                                       + ' because you told me ' + victim + ' was located here.', 'RescueBot')
                                    # Add the area to the list with searched areas
                                    if self._door['room_name'] not in self._searchedRooms:
                                        self._searchedRooms.append(self._door['room_name'])
                                    # Do not continue searching the rest of the area,
                                    # but start planning to rescue the victim
                                    self._phase = Phase.FIND_NEXT_GOAL

                            # Identify injured victim in the area
                            if 'healthy' not in victim and victim not in self._foundVictims:
                                self._recentVictim = victim
                                # Add the victim and the location to the corresponding dictionary
                                self._foundVictims.append(victim)
                                self._foundVictimLocations[victim] = {'location': info['location'], 'room': self._door['room_name'], 'obj_id': info['obj_id']}
                                # Communicate which victim the agent found,
                                # and ask the human whether to rescue the victim now or at a later stage
                                # TODO decide what to do about victim
                                if 'mild' in victim and not self._answered and not self._waiting:
                                    self._send_message('Found ' + victim + ' in ' + self._door['room_name'] +
                                                       '. Please decide whether to "Rescue together", "Rescue alone", or "Continue" searching. \n \n \
                                                        Important features to consider are: \n \
                                                        safe - victims rescued: ' + str(self._collectedVictims) + '\n \
                                                        explore - areas searched: area ' + str(self._searchedRooms).replace('area ', '') + '\n \
                                                        clock - extra time when rescuing alone: 15 seconds \n \
                                                        afstand - distance between us: ' + self._distanceToHuman, 'RescueBot')
                                    self._waiting = True
                                # TODO decide what to do about victim
                                if 'critical' in victim and not self._answered and not self._waiting:
                                    self._send_message('Found ' + victim + ' in ' + self._door['room_name'] +
                                                       '. Please decide whether to "Rescue" or "Continue" searching. \n\n \
                                                        Important features to consider are: \n \
                                                        explore - areas searched: area ' + str(self._searchedRooms).replace('area', '') + ' \n \
                                                        safe - victims rescued: ' + str(self._collectedVictims) + '\n \
                                                        afstand - distance between us: ' + self._distanceToHuman, 'RescueBot')
                                    self._waiting = True    
                    # Execute move actions to explore the area
                    return action, {}

                # Communicate that the agent did not find the target victim in the area,
                # while the human previously communicated the victim was located here
                # TODO modify trust
                if self._targetVictim in self._foundVictims and self._targetVictim not in self._victimsInRoom \
                        and self._foundVictimLocations[self._targetVictim]['room'] == self._door['room_name']:
                    self._send_message(self._targetVictim + ' not present in ' + str(self._door['room_name'])
                                       + ' because I searched the whole area without finding ' + self._targetVictim + '.', 'RescueBot')
                    # Remove the victim location from memory
                    self._foundVictimLocations.pop(self._targetVictim, None)
                    self._foundVictims.remove(self._targetVictim)
                    self._victimsInRoom = []
                    # Reset received messages (bug fix)
                    self.received_messages = []
                    self.received_messages_content = []
                # Add the area to the list of searched areas
                if self._door['room_name'] not in self._searchedRooms:
                    self._searchedRooms.append(self._door['room_name'])
                # Make a plan to rescue a found critically injured victim if the human decides so
                # TODO decide what to do about victim
                if self.received_messages_content and self.received_messages_content[-1] == 'Rescue' and 'critical' in self._recentVictim:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False
                    # Tell the human to come over and help carry the critically injured victim
                    if not state[{'is_human_agent': True}]:
                        self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry '
                                           + str(self._recentVictim) + ' together.', 'RescueBot')
                    # Tell the human to carry the critically injured victim together
                    if state[{'is_human_agent': True}]:
                        self._send_message('Lets carry ' + str(self._recentVictim) + ' together! \
                        Please wait until I moved on top of ' + str(self._recentVictim) + '.', 'RescueBot')
                    self._targetVictim = self._recentVictim
                    self._recentVictim = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Make a plan to rescue a found mildly injured victim together if the human decides so
                # TODO decide what to do about victim
                if self.received_messages_content and self.received_messages_content[-1] == 'Rescue together' and 'mild' in self._recentVictim:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False
                    # Tell the human to come over and help carry the mildly injured victim
                    if not state[{'is_human_agent': True}]:
                        self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry '
                                           + str(self._recentVictim) + ' together.', 'RescueBot')
                    # Tell the human to carry the mildly injured victim together
                    if state[{'is_human_agent': True}]:
                        self._send_message('Lets carry ' + str(self._recentVictim) + ' together! \
                        Please wait until I moved on top of ' + str(self._recentVictim) + '.', 'RescueBot')
                    self._targetVictim = self._recentVictim
                    self._recentVictim = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Make a plan to rescue the mildly injured victim alone,
                # if the human decides so and communicate this to the human
                if self.received_messages_content and self.received_messages_content[-1] == 'Rescue alone' and 'mild' in self._recentVictim:
                    self._send_message('Picking up ' + self._recentVictim + ' in '
                                       + self._door['room_name'] + '.', 'RescueBot')
                    self._rescue = 'alone'
                    self._answered = True
                    self._waiting = False
                    self._targetVictim = self._recentVictim
                    self._targetDropZone = self._remainingDropZones[self._targetVictim]
                    self._recentVictim = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Continue searching other areas if the human decides so
                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    self._answered = True
                    self._waiting = False
                    self._todo.append(self._recentVictim)
                    self._recentVictim = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Remain idle until the human communicates to the agent what to do with the found victim
                # TODO wait for how long
                if self.received_messages_content and self._waiting and self.received_messages_content[-1] != 'Rescue'\
                        and self.received_messages_content[-1] != 'Continue':
                    return None, {}
                # Find the next area to search when the agent is not waiting for an answer from the human
                # or occupied with rescuing a victim
                if not self._waiting and not self._rescue:
                    self._recentVictim = None
                    self._phase = Phase.FIND_NEXT_GOAL
                return Idle.__name__, {'duration_in_ticks': 25}

            if Phase.PLAN_PATH_TO_VICTIM == self._phase:
                # Plan the path to a found victim using its location
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._foundVictimLocations[self._targetVictim]['location']])
                # Follow the path to the found victim
                self._phase = Phase.FOLLOW_PATH_TO_VICTIM

            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                # Start searching for other victims if the human already rescued the target victim
                # TODO decide if we believe the human
                if self._targetVictim and self._targetVictim in self._collectedVictims:
                    self._phase = Phase.FIND_NEXT_GOAL
                # Otherwise, move towards the location of the found victim
                else:
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action is not None:
                        return action, {}
                    self._phase = Phase.TAKE_VICTIM

            if Phase.TAKE_VICTIM == self._phase:
                # Store all area tiles in a list
                room_tiles = [info['location'] for info in state.values()
                              if 'class_inheritance' in info
                              and 'AreaTile' in info['class_inheritance']
                              and 'room_name' in info
                              and info['room_name'] == self._foundVictimLocations[self._targetVictim]['room']]
                self._roomTiles = room_tiles
                objects = []
                # When the victim has to be carried by human and agent together,
                # check whether human has arrived at the victim's location
                for info in state.values():
                    # When the victim has to be carried by human and agent together,
                    # check whether human has arrived at the victim's location
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] \
                            and 'critical' in info['obj_id'] and info['location'] in self._roomTiles \
                            or 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] \
                            and 'mild' in info['obj_id'] and info['location'] in self._roomTiles \
                            and self._rescue == 'together' or self._targetVictim in self._foundVictims \
                            and self._targetVictim in self._todo and len(self._searchedRooms) == 0 \
                            and 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] \
                            and 'critical' in info['obj_id'] and info['location'] in self._roomTiles \
                            or self._targetVictim in self._foundVictims and self._targetVictim in self._todo \
                            and len(self._searchedRooms) == 0 and 'class_inheritance' in info \
                            and 'CollectableBlock' in info['class_inheritance'] and 'mild' in info['obj_id'] \
                            and info['location'] in self._roomTiles:
                        objects.append(info)
                        # Remain idle when the human has not arrived at the location
                        # TODO wait for how long
                        if self._humanName not in info['name']:
                            self._waiting = True
                            self._moving = False
                            return None, {}
                # Add the victim to the list of rescued victims when it has been picked up
                if len(objects) == 0 and 'critical' in self._targetVictim or len(objects) == 0 \
                        and 'mild' in self._targetVictim and self._rescue == 'together':
                    self._waiting = False
                    if self._targetVictim not in self._collectedVictims:
                        self._collectedVictims.append(self._targetVictim)
                    self._carryingTogether = True
                    # Determine the next victim to rescue or search
                    self._phase = Phase.FIND_NEXT_GOAL
                # When rescuing mildly injured victims alone, pick the victim up and plan the path to the drop zone
                if 'mild' in self._targetVictim and self._rescue == 'alone':
                    self._phase = Phase.PLAN_PATH_TO_DROPZONE
                    if self._targetVictim not in self._collectedVictims:
                        self._collectedVictims.append(self._targetVictim)
                    self._carrying = True
                    return CarryObject.__name__, {'object_id': self._foundVictimLocations[self._targetVictim]['obj_id'], 'human_name': self._humanName}

            if Phase.PLAN_PATH_TO_DROPZONE == self._phase:
                self._navigator.reset_full()
                # Plan the path to the drop zone
                self._navigator.add_waypoints([self._targetDropZone])
                # Follow the path to the drop zone
                self._phase = Phase.FOLLOW_PATH_TO_DROPZONE

            if Phase.FOLLOW_PATH_TO_DROPZONE == self._phase:
                # Communicate that the agent is transporting a mildly injured victim alone to the drop zone
                if 'mild' in self._targetVictim and self._rescue == 'alone':
                    self._send_message('Transporting ' + self._targetVictim + ' to the drop zone.', 'RescueBot')
                self._state_tracker.update(state)
                # Follow the path to the drop zone
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                # Drop the victim at the drop zone
                self._phase = Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                # Communicate that the agent delivered a mildly injured victim alone to the drop zone
                if 'mild' in self._targetVictim and self._rescue == 'alone':
                    self._send_message('Delivered ' + self._targetVictim + ' at the drop zone.', 'RescueBot')
                # Identify the next target victim to rescue
                self._phase = Phase.FIND_NEXT_GOAL
                self._rescue = None
                self._currentDoor = None
                self._carrying = False
                # Drop the victim on the correct location on the drop zone
                return Drop.__name__, {'human_name': self._humanName}

    def _process_messages(self, state):
        """
        Process incoming messages received from the team members
        """
        received_messages = {}
        # Create a dictionary with a list of received messages from each team member
        for member in self._teamMembers:
            received_messages[member] = []
        for msg in self.received_messages:
            for member in self._teamMembers:
                if msg.from_id == member:
                    received_messages[member].append(msg.content)
        # Check the content of the received messages
        for messages in received_messages.values():
            for msg in messages:
                # TODO decide if we believe the human
                # If a received message involves team members searching areas,
                # then add these areas to the memory of areas that have been explored
                if msg.startswith("Search:"):
                    area = 'area ' + msg.split()[-1]
                    if area not in self._searchedRooms:
                        self._searchedRooms.append(area)
                # If a received message involves team members finding victims,
                # then add these victims and their locations to memory
                if msg.startswith("Found:"):
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        reported_victim = ' '.join(msg.split()[1:4])
                    else:
                        reported_victim = ' '.join(msg.split()[1:5])
                    reported_location = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if reported_location not in self._searchedRooms:
                        self._searchedRooms.append(reported_location)
                    # Add the victim and its location to memory
                    if reported_victim not in self._foundVictims:
                        self._foundVictims.append(reported_victim)
                        self._foundVictimLocations[reported_victim] = {'room': reported_location}
                    if reported_victim in self._foundVictims and self._foundVictimLocations[reported_victim]['room'] != reported_location:
                        self._foundVictimLocations[reported_victim] = {'room': reported_location}
                    # Decide to help the human carry a found victim when the human's condition is 'weak'
                    if self._condition == 'weak':
                        self._rescue = 'together'
                    # Add the found victim to the to do list when the human's condition is not 'weak'
                    if 'mild' in reported_victim and self._condition != 'weak':
                        self._todo.append(reported_victim)
                # If a received message involves team members rescuing victims,
                # then add these victims and their locations to memory
                if msg.startswith('Collect:'):
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        collect_victim = ' '.join(msg.split()[1:4])
                    else:
                        collect_victim = ' '.join(msg.split()[1:5])
                    collect_location = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if collect_location not in self._searchedRooms:
                        self._searchedRooms.append(collect_location)
                    # Add the victim and location to the memory of found victims
                    if collect_victim not in self._foundVictims:
                        self._foundVictims.append(collect_victim)
                        self._foundVictimLocations[collect_victim] = {'room': collect_location}
                    if collect_victim in self._foundVictims and self._foundVictimLocations[collect_victim]['room'] != collect_location:
                        self._foundVictimLocations[collect_victim] = {'room': collect_location}
                    # Add the victim to the memory of rescued victims when the human's condition is not weak
                    if self._condition != 'weak' and collect_victim not in self._collectedVictims:
                        self._collectedVictims.append(collect_victim)
                    # Decide to help the human carry the victim together when the human's condition is weak
                    if self._condition == 'weak':
                        self._rescue = 'together'
                # If a received message involves team members asking for help with removing obstacles,
                # then add their location to memory and come over
                if msg.startswith('Remove:'):
                    # Come over immediately when the agent is not carrying a victim
                    if not self._carrying:
                        # Identify at which location the human needs help
                        area = 'area ' + msg.split()[-1]
                        self._door = state.get_room_doors(area)[0]
                        self._doormat = state.get_room(area)[-1]['doormat']
                        if area in self._searchedRooms:
                            self._searchedRooms.remove(area)
                        # Clear received messages (bug fix)
                        self.received_messages = []
                        self.received_messages_content = []
                        self._moving = True
                        self._remove = True
                        if self._waiting and self._recentVictim:
                            self._todo.append(self._recentVictim)
                        self._waiting = False
                        # Let the human know that the agent is coming over to help
                        self._send_message('Moving to ' + str(self._door['room_name'])
                                           + ' to help you remove an obstacle.', 'RescueBot')
                        # Plan the path to the relevant area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Come over to help after dropping a victim that is currently being carried by the agent
                    else:
                        area = 'area ' + msg.split()[-1]
                        self._send_message('Will come to ' + area + ' after dropping '
                                           + self._targetVictim + '.', 'RescueBot')
            # Store the current location of the human in memory
            if messages and messages[-1].split()[-1] in ['1', '2', '3', '4', '5', '6', '7',
                                                         '8', '9', '10', '11', '12', '13', '14']:
                self._humanLocation = int(messages[-1].split()[-1])

    def _load_beliefs(self):
        """
        Loads trust belief values if agent already collaborated with human before,
        otherwise trust belief values are initialized using default values.
        """
        # Set a default trust values
        default = 0.5
        # Load file with all trust beliefs
        trust_file_header = []
        with open(self._folder+'/beliefs/localAllTrustBeliefs.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            for row in reader:
                if not trust_file_header:
                    trust_file_header = row
                    continue
                # Retrieve trust values 
                if row:
                    # Check if agent already collaborated with this human before,
                    # if yes: load the corresponding trust values,
                    # if no: use the default trust values
                    name = row[0]
                    competence = default
                    willingness = default
                    if name == self._humanName:
                        # load the corresponding trust values
                        competence = float(row[1])
                        willingness = float(row[2])
                    self._trustBeliefs[self._humanName] = {'competence': competence, 'willingness': willingness}

    def _send_message(self, content, sender):
        """
        send messages from agent to other team members
        """
        msg = Message(content=content, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
            self.send_message(msg)
            self._sendMessages.append(msg.content)
        # Sending the hidden score message (DO NOT REMOVE)
        if 'Our score is' in msg.content:
            self.send_message(msg)

    def _get_closest_room(self, state, objs, current_door):
        """
        Calculate which area is closest to the agent's location
        """
        agent_location = state[self.agent_id]['location']
        locations = {}
        for obj in objs:
            locations[obj] = state.get_room_doors(obj)[0]['location']
        distances = {}
        for room, location in locations.items():
            if current_door is not None:
                distances[room] = utils.get_distance(current_door, location)
            if current_door is None:
                distances[room] = utils.get_distance(agent_location, location)

        return min(distances, key=distances.get)
