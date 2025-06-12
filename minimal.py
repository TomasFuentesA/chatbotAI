# suppress warnings
import warnings

warnings.filterwarnings("ignore")

# import libraries
import argparse
from together import Together
import textwrap
import os
from datetime import datetime
import json
import random
import math

# D&D Character Constants
CHARACTER_CLASSES = ["Warrior", "Mage", "Rogue", "Cleric"]
RACES = ["Human", "Elf", "Dwarf", "Halfling"]
BACKGROUNDS = ["Noble", "Sage", "Criminal", "Folk Hero"]
ALIGNMENTS = ["Lawful Good", "Neutral Good", "Chaotic Good", 
              "Lawful Neutral", "True Neutral", "Chaotic Neutral",
              "Lawful Evil", "Neutral Evil", "Chaotic Evil"]

# Game Constants
XP_PER_LEVEL = {
    1: 0,
    2: 300,
    3: 900,
    4: 2700,
    5: 6500,
    6: 14000,
    7: 23000,
    8: 34000,
    9: 48000,
    10: 64000
}

WEAPON_TYPES = {
    "Warrior": ["Longsword", "Battleaxe", "Warhammer"],
    "Mage": ["Staff", "Dagger", "Wand"],
    "Rogue": ["Shortsword", "Rapier", "Shortbow"],
    "Cleric": ["Mace", "Warhammer", "Crossbow"]
}

# Item Constants
ITEM_TYPES = {
    "weapon": ["Sword", "Axe", "Bow", "Staff", "Dagger"],
    "armor": ["Leather", "Chain", "Plate", "Robe"],
    "potion": ["Health", "Mana", "Strength", "Invisibility"],
    "scroll": ["Fireball", "Lightning Bolt", "Heal", "Teleport"]
}

def load_dnd_content(file_path="data/d&d.txt", additional_files=None):
    """Load D&D content from main file and optional additional files."""
    content = {}
    
    # Load main D&D file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content['main'] = file.read()
    except FileNotFoundError:
        print(f"Warning: {file_path} not found.")
        content['main'] = ""

    # Load additional files if provided
    if additional_files:
        for file_path in additional_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content[os.path.basename(file_path)] = file.read()
            except FileNotFoundError:
                print(f"Warning: {file_path} not found.")

    return content

def summarize_concepts(content, max_length=100):
    """Create bite-sized summaries of D&D concepts."""
    summaries = {}
    
    for source, text in content.items():
        if text:  # Only summarize non-empty content
            prompt = f"Summarize the following D&D concept in {max_length} characters or less:\n{text}"
            summary = prompt_llm(prompt)
            summaries[source] = summary
    
    return summaries

def save_summaries(summaries, timestamp=None):
    """Save the summaries to a timestamped file."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{timestamp}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== D&D Concept Summaries ===\n\n")
        for source, summary in summaries.items():
            f.write(f"Source: {source}\n")
            f.write(f"Summary: {summary}\n")
            f.write("-" * 50 + "\n")
    
    return output_file

## FUNCTION 1: This Allows Us to Prompt the AI MODEL
# -------------------------------------------------
def prompt_llm(prompt, with_linebreak=False):
    # This function allows us to prompt an LLM via the Together API

    # model
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    output = response.choices[0].message.content

    if with_linebreak:
        # Wrap the output
        wrapped_output = textwrap.fill(output, width=50)
        return wrapped_output
    else:
        return output

def build_character():
    """Build a D&D character with user input and random generation."""
    print("\n=== Character Creation ===")
    
    # Basic Information
    name = input("Enter your character's name: ")
    
    # Race Selection
    print("\nAvailable races:")
    for i, race in enumerate(RACES, 1):
        print(f"{i}. {race}")
    race_choice = int(input("Choose your race (1-4): ")) - 1
    race = RACES[race_choice]
    
    # Class Selection
    print("\nAvailable classes:")
    for i, char_class in enumerate(CHARACTER_CLASSES, 1):
        print(f"{i}. {char_class}")
    class_choice = int(input("Choose your class (1-4): ")) - 1
    char_class = CHARACTER_CLASSES[class_choice]
    
    # Background Selection
    print("\nAvailable backgrounds:")
    for i, background in enumerate(BACKGROUNDS, 1):
        print(f"{i}. {background}")
    background_choice = int(input("Choose your background (1-4): ")) - 1
    background = BACKGROUNDS[background_choice]
    
    # Alignment Selection
    print("\nAvailable alignments:")
    for i, alignment in enumerate(ALIGNMENTS, 1):
        print(f"{i}. {alignment}")
    alignment_choice = int(input("Choose your alignment (1-9): ")) - 1
    alignment = ALIGNMENTS[alignment_choice]
    
    # Generate random stats (3d6 method)
    stats = {
        "Strength": sum(random.randint(1, 6) for _ in range(3)),
        "Dexterity": sum(random.randint(1, 6) for _ in range(3)),
        "Constitution": sum(random.randint(1, 6) for _ in range(3)),
        "Intelligence": sum(random.randint(1, 6) for _ in range(3)),
        "Wisdom": sum(random.randint(1, 6) for _ in range(3)),
        "Charisma": sum(random.randint(1, 6) for _ in range(3))
    }
    
    # Generate random HP based on class and constitution
    base_hp = {
        "Warrior": 10,
        "Mage": 6,
        "Rogue": 8,
        "Cleric": 8
    }
    constitution_modifier = (stats["Constitution"] - 10) // 2
    hp = base_hp[char_class] + constitution_modifier
    
    # Create character dictionary
    character = {
        "name": name,
        "race": race,
        "class": char_class,
        "background": background,
        "alignment": alignment,
        "level": 1,
        "experience": 0,
        "hit_points": max(1, hp),  # Ensure at least 1 HP
        "stats": stats,
        "inventory": [],
        "skills": [],
        "proficiencies": []
    }
    
    # Generate character backstory using LLM
    backstory_prompt = f"Create a brief backstory for a {race} {char_class} with a {background} background who is {alignment}. Keep it under 100 words."
    character["backstory"] = prompt_llm(backstory_prompt)
    
    return character

def save_character(character, timestamp=None):
    """Save character information to a file."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"character_{timestamp}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== D&D Character Sheet ===\n\n")
        f.write(f"Name: {character['name']}\n")
        f.write(f"Race: {character['race']}\n")
        f.write(f"Class: {character['class']}\n")
        f.write(f"Background: {character['background']}\n")
        f.write(f"Alignment: {character['alignment']}\n")
        f.write(f"Level: {character['level']}\n")
        f.write(f"Experience: {character['experience']}\n")
        f.write(f"Hit Points: {character['hit_points']}\n\n")
        
        f.write("=== Ability Scores ===\n")
        for stat, value in character['stats'].items():
            modifier = (value - 10) // 2
            modifier_str = f"+{modifier}" if modifier >= 0 else str(modifier)
            f.write(f"{stat}: {value} ({modifier_str})\n")
        
        f.write("\n=== Backstory ===\n")
        f.write(character['backstory'])
    
    return output_file

def roll_dice(dice_type="d20", num_dice=1):
    """Roll dice for combat and skill checks."""
    try:
        num, sides = map(int, dice_type.lower().split('d'))
        return sum(random.randint(1, sides) for _ in range(num * num_dice))
    except ValueError:
        return 0

def calculate_attack(character, weapon_type="melee"):
    """Calculate attack roll and damage."""
    # Get relevant stat modifier
    if weapon_type == "melee":
        stat_mod = (character['stats']['Strength'] - 10) // 2
    else:  # ranged
        stat_mod = (character['stats']['Dexterity'] - 10) // 2
    
    # Calculate attack bonus
    proficiency_bonus = math.floor((character['level'] - 1) / 4) + 2
    attack_bonus = stat_mod + proficiency_bonus
    
    # Roll attack
    attack_roll = roll_dice("d20") + attack_bonus
    
    # Calculate damage
    weapon_damage = roll_dice("d8")  # Basic weapon damage
    total_damage = weapon_damage + stat_mod
    
    return {
        "attack_roll": attack_roll,
        "damage": max(1, total_damage),  # Minimum 1 damage
        "attack_bonus": attack_bonus
    }

def resolve_combat(character, enemy):
    """Handle combat between character and enemy."""
    print(f"\n=== Combat: {character['name']} vs {enemy['name']} ===")
    
    # Initialize combat
    character_hp = character['hit_points']
    enemy_hp = enemy['hit_points']
    round_num = 1
    
    while character_hp > 0 and enemy_hp > 0:
        print(f"\nRound {round_num}")
        
        # Character's turn
        print(f"\n{character['name']}'s turn:")
        attack = calculate_attack(character)
        print(f"Attack roll: {attack['attack_roll']}")
        
        if attack['attack_roll'] >= enemy['armor_class']:
            enemy_hp -= attack['damage']
            print(f"Hit! Dealt {attack['damage']} damage to {enemy['name']}")
        else:
            print("Miss!")
        
        # Check if enemy is defeated
        if enemy_hp <= 0:
            print(f"\n{enemy['name']} has been defeated!")
            return True
        
        # Enemy's turn
        print(f"\n{enemy['name']}'s turn:")
        enemy_attack = calculate_attack(enemy)
        print(f"Attack roll: {enemy_attack['attack_roll']}")
        
        if enemy_attack['attack_roll'] >= character['armor_class']:
            character_hp -= enemy_attack['damage']
            print(f"Hit! Dealt {enemy_attack['damage']} damage to {character['name']}")
        else:
            print("Miss!")
        
        # Check if character is defeated
        if character_hp <= 0:
            print(f"\n{character['name']} has been defeated!")
            return False
        
        round_num += 1

def generate_quest(character_level, character_class):
    """Generate a random quest appropriate for the character."""
    prompt = f"""Create a D&D quest for a level {character_level} {character_class}. 
    Include:
    1. A brief description
    2. Main objective
    3. Potential rewards
    4. Difficulty level
    Keep it under 150 words."""
    
    quest = prompt_llm(prompt)
    
    # Generate random rewards
    gold = random.randint(10, 50) * character_level
    xp = random.randint(50, 200) * character_level
    
    return {
        "description": quest,
        "rewards": {
            "gold": gold,
            "xp": xp
        },
        "completed": False
    }

def track_quest_progress(character, quest):
    """Track progress and completion of quests."""
    if not quest['completed']:
        # Add rewards to character
        character['gold'] = character.get('gold', 0) + quest['rewards']['gold']
        gain_experience(character, quest['rewards']['xp'])
        quest['completed'] = True
        return True
    return False

def level_up(character):
    """Handle character leveling up."""
    if character['level'] >= 10:  # Max level
        return False
    
    character['level'] += 1
    
    # Update HP
    base_hp = {
        "Warrior": 10,
        "Mage": 6,
        "Rogue": 8,
        "Cleric": 8
    }
    constitution_modifier = (character['stats']['Constitution'] - 10) // 2
    hp_increase = base_hp[character['class']] + constitution_modifier
    character['hit_points'] += max(1, hp_increase)
    
    # Update proficiency bonus
    character['proficiency_bonus'] = math.floor((character['level'] - 1) / 4) + 2
    
    # Generate level up message using LLM
    prompt = f"""Create a brief level up message for a {character['race']} {character['class']} 
    reaching level {character['level']}. Include new abilities gained. Keep it under 100 words."""
    character['level_up_message'] = prompt_llm(prompt)
    
    return True

def gain_experience(character, amount):
    """Add experience and check for level up."""
    character['experience'] += amount
    
    # Check for level up
    current_level = character['level']
    while (current_level < 10 and 
           character['experience'] >= XP_PER_LEVEL.get(current_level + 1, float('inf'))):
        level_up(character)
        current_level = character['level']
    
    return current_level > character['level']  # Return True if leveled up

def add_item(character, item):
    """Add item to character inventory."""
    if 'inventory' not in character:
        character['inventory'] = []
    
    # Check if item is a weapon and character can use it
    if item.get('type') == 'weapon':
        if item['name'] not in WEAPON_TYPES.get(character['class'], []):
            print(f"Warning: {character['class']} cannot use {item['name']}")
            return False
    
    character['inventory'].append(item)
    return True

def use_item(character, item_name):
    """Use an item from inventory."""
    # Find item in inventory
    item = next((i for i in character['inventory'] if i['name'] == item_name), None)
    if not item:
        print(f"Item {item_name} not found in inventory")
        return False
    
    # Apply item effects
    if item['type'] == 'potion':
        if 'Health' in item['name']:
            character['hit_points'] = min(
                character['hit_points'] + item['effect'],
                character['max_hit_points']
            )
        elif 'Mana' in item['name']:
            character['mana'] = min(
                character.get('mana', 0) + item['effect'],
                character.get('max_mana', 0)
            )
    
    # Remove consumable items
    if item['type'] in ['potion', 'scroll']:
        character['inventory'].remove(item)
    
    return True

def save_game_state(character, current_quest=None):
    """Save complete game state including character and current quest."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    save_data = {
        "character": character,
        "current_quest": current_quest,
        "timestamp": timestamp
    }
    
    output_file = os.path.join(output_dir, f"save_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)
    
    return output_file

def load_game_state(save_file):
    """Load a saved game state."""
    try:
        with open(save_file, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        return save_data['character'], save_data['current_quest']
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading save file: {e}")
        return None, None

def generate_npc(character_level):
    """Generate a random NPC with appropriate stats and personality."""
    # Generate NPC type
    npc_types = ["Merchant", "Guard", "Mage", "Rogue", "Cleric"]
    npc_type = random.choice(npc_types)
    
    # Generate NPC stats
    stats = {
        "Strength": roll_dice("3d6"),
        "Dexterity": roll_dice("3d6"),
        "Constitution": roll_dice("3d6"),
        "Intelligence": roll_dice("3d6"),
        "Wisdom": roll_dice("3d6"),
        "Charisma": roll_dice("3d6")
    }
    
    # Generate NPC personality using LLM
    prompt = f"""Create a brief personality description for a {npc_type} NPC in a D&D game.
    Include their demeanor, goals, and a unique trait. Keep it under 100 words."""
    personality = prompt_llm(prompt)
    
    return {
        "name": f"{npc_type} NPC",
        "type": npc_type,
        "level": max(1, character_level - 1 + random.randint(-1, 1)),
        "stats": stats,
        "personality": personality,
        "inventory": [],
        "quests": []
    }

def handle_dialogue(character, npc):
    """Manage dialogue interactions with NPCs."""
    # Generate dialogue options based on NPC type and character
    prompt = f"""Create a dialogue interaction between a {character['race']} {character['class']} 
    and a {npc['type']}. Include 3-4 dialogue options for the player. Keep it under 200 words."""
    
    dialogue = prompt_llm(prompt)
    
    # Generate NPC response based on player choice
    def get_npc_response(choice):
        response_prompt = f"""The {npc['type']} responds to: "{choice}"
        Consider their personality: {npc['personality']}
        Keep the response under 50 words."""
        return prompt_llm(response_prompt)
    
    return {
        "dialogue": dialogue,
        "get_response": get_npc_response
    }

def generate_location(character_level):
    """Generate a random location for adventures."""
    location_types = [
        "Dungeon", "Forest", "Town", "Castle", "Ruins",
        "Cave", "Temple", "Swamp", "Mountain", "Desert"
    ]
    
    location_type = random.choice(location_types)
    
    prompt = f"""Create a brief description of a {location_type} location for a D&D adventure.
    Include:
    1. General atmosphere
    2. Notable features
    3. Potential dangers
    4. Possible rewards
    Keep it under 150 words."""
    
    description = prompt_llm(prompt)
    
    return {
        "type": location_type,
        "description": description,
        "level": character_level,
        "encounters": [],
        "treasures": []
    }

def generate_encounter(character_level, location):
    """Generate random encounters based on location and level."""
    encounter_types = [
        "Combat", "Puzzle", "Social", "Exploration",
        "Trap", "Treasure", "Quest"
    ]
    
    encounter_type = random.choice(encounter_types)
    
    prompt = f"""Create a {encounter_type} encounter in a {location['type']} for a level {character_level} character.
    Include:
    1. Setup
    2. Challenge
    3. Possible outcomes
    Keep it under 150 words."""
    
    encounter = prompt_llm(prompt)
    
    # Generate rewards
    rewards = {
        "gold": random.randint(10, 50) * character_level,
        "xp": random.randint(50, 200) * character_level,
        "items": []
    }
    
    # Add random item reward
    if random.random() < 0.3:  # 30% chance for item
        item_type = random.choice(list(ITEM_TYPES.keys()))
        item_name = random.choice(ITEM_TYPES[item_type])
        rewards["items"].append({
            "name": item_name,
            "type": item_type,
            "effect": random.randint(1, 10)
        })
    
    return {
        "type": encounter_type,
        "description": encounter,
        "rewards": rewards,
        "completed": False
    }

def calculate_skill_modifiers(character):
    """Calculate all skill modifiers based on stats and proficiencies."""
    skills = {
        "Athletics": "Strength",
        "Acrobatics": "Dexterity",
        "Stealth": "Dexterity",
        "Arcana": "Intelligence",
        "History": "Intelligence",
        "Investigation": "Intelligence",
        "Nature": "Intelligence",
        "Religion": "Intelligence",
        "Animal Handling": "Wisdom",
        "Insight": "Wisdom",
        "Medicine": "Wisdom",
        "Perception": "Wisdom",
        "Survival": "Wisdom",
        "Deception": "Charisma",
        "Intimidation": "Charisma",
        "Performance": "Charisma",
        "Persuasion": "Charisma"
    }
    
    modifiers = {}
    for skill, stat in skills.items():
        stat_mod = (character['stats'][stat] - 10) // 2
        proficiency = 2 if skill in character.get('proficiencies', []) else 0
        modifiers[skill] = stat_mod + proficiency
    
    return modifiers

def update_character_sheet(character):
    """Update and format the character sheet with all current information."""
    # Calculate skill modifiers
    skill_modifiers = calculate_skill_modifiers(character)
    
    # Update character sheet
    character['skill_modifiers'] = skill_modifiers
    
    # Generate updated character description using LLM
    prompt = f"""Create a brief character summary for a level {character['level']} {character['race']} {character['class']}.
    Include their notable achievements and current status. Keep it under 100 words."""
    character['summary'] = prompt_llm(prompt)
    
    return character

if __name__ == "__main__":
    # args on which to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--api_key", type=str, default=None)
    parser.add_argument("-a", "--additional_files", nargs="+", help="Additional files to load")
    parser.add_argument("-c", "--create_character", action="store_true", help="Create a new character")
    parser.add_argument("-l", "--load_save", type=str, help="Load a saved game file")
    parser.add_argument("-n", "--new_game", action="store_true", help="Start a new game")
    args = parser.parse_args()

    # Get Client for your LLMs
    client = Together(api_key=args.api_key)

    if args.load_save:
        # Load saved game
        character, current_quest = load_game_state(args.load_save)
        if character:
            print(f"\nLoaded character: {character['name']}")
            print(f"Level {character['level']} {character['race']} {character['class']}")
        else:
            print("Failed to load saved game")
            exit(1)
    elif args.create_character:
        # Create new character
        character = build_character()
        output_file = save_character(character)
        print(f"\nCharacter sheet has been saved to: {output_file}")
        
        # Display character sheet
        print("\n=== Your Character ===")
        print(json.dumps(character, indent=2))
    elif args.new_game:
        # Start new game
        character = build_character()
        current_quest = None
        current_location = None
        
        print("\n=== Starting New Game ===")
        print(f"Welcome, {character['name']}!")
        
        while True:
            print("\nWhat would you like to do?")
            print("1. View character sheet")
            print("2. Start a new quest")
            print("3. Explore a new location")
            print("4. Talk to an NPC")
            print("5. Use an item")
            print("6. Save game")
            print("7. Quit")
            
            choice = input("\nEnter your choice (1-7): ")
            
            if choice == "1":
                character = update_character_sheet(character)
                print("\n=== Character Sheet ===")
                print(json.dumps(character, indent=2))
            
            elif choice == "2":
                current_quest = generate_quest(character['level'], character['class'])
                print("\n=== New Quest ===")
                print(current_quest['description'])
                print(f"\nRewards: {current_quest['rewards']['gold']} gold, {current_quest['rewards']['xp']} XP")
            
            elif choice == "3":
                current_location = generate_location(character['level'])
                print(f"\n=== New Location: {current_location['type']} ===")
                print(current_location['description'])
                
                # Generate random encounter
                encounter = generate_encounter(character['level'], current_location)
                print(f"\n=== Encounter: {encounter['type']} ===")
                print(encounter['description'])
                
                # Handle encounter completion
                if encounter['type'] == 'Combat':
                    enemy = generate_npc(character['level'])
                    victory = resolve_combat(character, enemy)
                    if victory:
                        print("\n=== Rewards ===")
                        print(f"Gold: {encounter['rewards']['gold']}")
                        print(f"XP: {encounter['rewards']['xp']}")
                        if encounter['rewards']['items']:
                            for item in encounter['rewards']['items']:
                                add_item(character, item)
                                print(f"Found: {item['name']}")
                        gain_experience(character, encounter['rewards']['xp'])
                        character['gold'] = character.get('gold', 0) + encounter['rewards']['gold']
            
            elif choice == "4":
                npc = generate_npc(character['level'])
                print(f"\n=== Meeting {npc['name']} ===")
                print(npc['personality'])
                
                dialogue = handle_dialogue(character, npc)
                print("\n=== Dialogue ===")
                print(dialogue['dialogue'])
                
                response = input("\nEnter your response: ")
                npc_response = dialogue['get_response'](response)
                print(f"\n{npc['name']}: {npc_response}")
            
            elif choice == "5":
                if not character.get('inventory'):
                    print("\nYour inventory is empty!")
                else:
                    print("\n=== Inventory ===")
                    for i, item in enumerate(character['inventory'], 1):
                        print(f"{i}. {item['name']} ({item['type']})")
                    
                    item_choice = int(input("\nChoose an item to use (number): ")) - 1
                    if 0 <= item_choice < len(character['inventory']):
                        item = character['inventory'][item_choice]
                        use_item(character, item['name'])
                        print(f"\nUsed {item['name']}")
            
            elif choice == "6":
                save_file = save_game_state(character, current_quest)
                print(f"\nGame saved to: {save_file}")
            
            elif choice == "7":
                save_game_state(character, current_quest)
                print("\nGame saved. Goodbye!")
                break
            
            else:
                print("\nInvalid choice. Please try again.")
    else:
        # Load D&D content
        content = load_dnd_content(additional_files=args.additional_files)
        
        # Generate summaries
        summaries = summarize_concepts(content)
        
        # Save summaries
        output_file = save_summaries(summaries)
        print(f"\nSummaries have been saved to: {output_file}")
        
        # Display summaries
        print("\nGenerated Summaries:")
        print("-" * 50)
        for source, summary in summaries.items():
            print(f"\nSource: {source}")
            print(f"Summary: {summary}")
            print("-" * 50)