import discord
import datetime
import asyncio
from dotenv import dotenv_values
import pandas as pd
from datetime import datetime, timedelta, timezone
import openai
import schedule
from openai import OpenAI
from discord.ui import View, Modal, Button, TextInput
from discord.ext import commands
import json
import os

config = dotenv_values(".env")
API_TOKEN = config["BOT_TOKEN"]
GPT_TOKEN = config["GPT_TOKEN"]

openai.api_key = GPT_TOKEN
openai_client = OpenAI(api_key=GPT_TOKEN)

# Set up intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.reactions = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Global dictionaries for storing messages, reactions, reply chains, and conversation chains
GLOBAL_MESSAGES = {}  # {guild_id: {channel_id: [(message_id, author_id, timestamp, message_content), ...]}}
GLOBAL_REACTIONS = {}  # {guild_id: {message_id: [(emoji, count), ...]}}
GLOBAL_REPLY_CHAINS = {}  # {guild_id: {message_id: [list_of_reply_message_ids]}}
GLOBAL_CONVERSATION_CHAINS = {}  # {guild_id: [[message_id_1, message_id_2, message_id_3], ...]}
COLLABORATIVE_MESSAGES = {} # {guild_id: [message_id_1, message_id_2, message_id_3,...], ...}
# Store the weekly reports to avoid recalculating them on button presses
WEEKLY_REPORTS = {}  # {guild_id: "Formatted weekly report string"}

# Define the backup directory
BACKUP_DIR = "backups"

def backup_global_dictionaries():
    """Backup all global dictionaries to JSON files"""
    # Create backup directory if it doesn't exist
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    # Dictionary mapping global variables to their backup filenames
    backup_mapping = {
        GLOBAL_MESSAGES: "global_messages.json",
        GLOBAL_REACTIONS: "global_reactions.json",
        GLOBAL_REPLY_CHAINS: "reply_chains.json",
        GLOBAL_CONVERSATION_CHAINS: "conversation_chains.json",
        COLLABORATIVE_MESSAGES: "collaborative_messages.json",
        WEEKLY_REPORTS: "weekly_reports.json",
    }
    
    for dict_obj, filename in backup_mapping.items():
        filepath = os.path.join(BACKUP_DIR, filename)
        try:
            # Convert all non-string dictionary keys to strings for JSON serialization
            serializable_dict = {}
            for key, value in dict_obj.items():
                str_key = str(key)
                if isinstance(value, dict):
                    inner_dict = {}
                    for k, v in value.items():
                        inner_dict[str(k)] = v
                    serializable_dict[str_key] = inner_dict
                else:
                    serializable_dict[str_key] = value
                    
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error backing up {filename}: {str(e)}")

def load_global_dictionaries():
    """Load global dictionaries from JSON files"""
    # Dictionary mapping filenames to global variables
    load_mapping = {
        "global_messages.json": GLOBAL_MESSAGES,
        "global_reactions.json": GLOBAL_REACTIONS,
        "reply_chains.json": GLOBAL_REPLY_CHAINS,
        "conversation_chains.json": GLOBAL_CONVERSATION_CHAINS,
        "collaborative_messages.json": COLLABORATIVE_MESSAGES,
        "weekly_reports.json": WEEKLY_REPORTS,
    }
    
    for filename, dict_obj in load_mapping.items():
        filepath = os.path.join(BACKUP_DIR, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Convert string keys back to integers where needed
                converted_data = {}
                for key, value in data.items():
                    try:
                        int_key = int(key)
                        if isinstance(value, dict):
                            inner_dict = {}
                            for k, v in value.items():
                                try:
                                    inner_dict[int(k)] = v
                                except ValueError:
                                    inner_dict[k] = v
                            converted_data[int_key] = inner_dict
                        else:
                            converted_data[int_key] = value
                    except ValueError:
                        converted_data[key] = value
                        
                dict_obj.clear()
                dict_obj.update(converted_data)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

@bot.event
async def on_guild_join(guild: discord.Guild):
    """
    Fires when the bot joins a new server (guild).
    Gathers messages from the past 7 days, reactions, reply chains, and combines isolated blocks of messages.
    """
    print(f"Bot added to the guild: {guild.name} (ID: {guild.id})")

    # Ensure the guild_id has its own entry in global dictionaries
    if guild.id not in GLOBAL_MESSAGES:
        GLOBAL_MESSAGES[guild.id] = {}
    if guild.id not in GLOBAL_REACTIONS:
        GLOBAL_REACTIONS[guild.id] = {}
    if guild.id not in GLOBAL_REPLY_CHAINS:
        GLOBAL_REPLY_CHAINS[guild.id] = {}
    if guild.id not in GLOBAL_CONVERSATION_CHAINS:
        GLOBAL_CONVERSATION_CHAINS[guild.id] = []
    if guild.id not in COLLABORATIVE_MESSAGES:
        COLLABORATIVE_MESSAGES[guild.id] = []

    # Calculate the cutoff date/time (7 days ago)
    after_date = datetime.now(timezone.utc) - timedelta(days=7)

    for channel in guild.text_channels:
        if not channel.permissions_for(guild.me).read_message_history:
            print(f"Skipping channel '{channel.name}' - missing Read Message History permission.")
            continue

        if channel.id not in GLOBAL_MESSAGES[guild.id]:
            GLOBAL_MESSAGES[guild.id][channel.id] = []

        print(f"Collecting messages from channel: #{channel.name}")

        try:
            async for msg in channel.history(limit=None, after=after_date):
                # Store message details
                if msg.content == '':
                    continue
                GLOBAL_MESSAGES[guild.id][channel.id].append(
                    (msg.id, msg.author.id, msg.created_at.isoformat(), msg.content)
                )

                # Check if the message is a reply
                if msg.reference and msg.reference.message_id:
                    replied_to_id = msg.reference.message_id

                    # Track direct reply chains
                    if replied_to_id not in GLOBAL_REPLY_CHAINS[guild.id]:
                        GLOBAL_REPLY_CHAINS[guild.id][replied_to_id] = []
                    GLOBAL_REPLY_CHAINS[guild.id][replied_to_id].append(msg.id)

                    # Build full conversation chain and store it
                    conversation_chain = build_conversation_chain(msg.id, guild.id)
                    GLOBAL_CONVERSATION_CHAINS[guild.id].append(conversation_chain)

        except discord.Forbidden:
            print(f"Forbidden: No permissions to read messages in {channel.name}.")
        except discord.HTTPException as e:
            print(f"HTTP Exception in {channel.name}: {e}")

    total_msgs = sum(len(msg_list) for msg_list in GLOBAL_MESSAGES[guild.id].values())
    print(f"Finished fetching messages for {guild.name}. Total messages collected: {total_msgs}")

    # Combine consecutive isolated messages into blocks
    combine_isolated_message_blocks(guild.id)

    # Gather reactions after fetching messages
    await gather_reactions_for_guild(guild.id)
    print("Finished gathering all reactions.")
    print_reaction_summary(guild.id)


@bot.event
async def on_message(message):
    """
    Fires when a new message is sent in any channel the bot can see.
    Adds the message to GLOBAL_MESSAGES, combines it if isolated, and updates reply tracking if it's a reply.
    """
    if message.author == bot.user:
        return

    guild_id = message.guild.id
    channel_id = message.channel.id

    # Ensure the server and channel keys exist
    if guild_id not in GLOBAL_MESSAGES:
        GLOBAL_MESSAGES[guild_id] = {}
    if channel_id not in GLOBAL_MESSAGES[guild_id]:
        GLOBAL_MESSAGES[guild_id][channel_id] = []
    if guild_id not in GLOBAL_REPLY_CHAINS:
        GLOBAL_REPLY_CHAINS[guild_id] = {}

    # Get the timestamp and message details
    message_entry = (message.id, message.author.id, message.created_at.isoformat(), message.content)
    if message.content == '':
        return

    # Check if the message should be combined with the last message in the channel
    if should_combine_with_last_message(guild_id, channel_id, message_entry):
        # Combine the current message with the last one
        combine_with_last_message(guild_id, channel_id, message_entry)
    else:
        # Add the new message as a separate entry
        GLOBAL_MESSAGES[guild_id][channel_id].append(message_entry)

    # Check if the message is a reply
    if message.reference and message.reference.message_id:
        replied_to_id = message.reference.message_id

        # Track the reply in GLOBAL_REPLY_CHAINS
        if replied_to_id not in GLOBAL_REPLY_CHAINS[guild_id]:
            GLOBAL_REPLY_CHAINS[guild_id][replied_to_id] = []
        GLOBAL_REPLY_CHAINS[guild_id][replied_to_id].append(message.id)

    print(
        f"At {message.created_at} in '{message.channel.name}' (Server: {message.guild.name}), "
        f"{message.author} said: {message.content} (ID: {message.id})"
    )

    if message.content == "dEbUg":
        print(f'Global Messages: {str(GLOBAL_MESSAGES)}')
        print('-------')
        print('-------')
        print(f'Global Reactions: {str(GLOBAL_REACTIONS)}')
        print('-------')
        print('-------')
        print(calculate_influence_scores())
        await display_all_guild_topics()

    # This line is crucial - it allows both commands and events to work
    await bot.process_commands(message)

def update_reaction(guild_id, message_id, emoji, change):
    """
    Updates the reaction count for a message in GLOBAL_REACTIONS.

    Args:
        guild_id: The ID of the guild (server).
        message_id: The ID of the message.
        emoji: The emoji of the reaction.
        change: +1 if a reaction was added, -1 if it was removed.
    """
    if message_id not in GLOBAL_REACTIONS[guild_id]:
        GLOBAL_REACTIONS[guild_id][message_id] = []

    # Find the existing emoji in the list or add it if it's not there
    for i, (existing_emoji, count) in enumerate(GLOBAL_REACTIONS[guild_id][message_id]):
        if existing_emoji == emoji:
            # Update the count
            GLOBAL_REACTIONS[guild_id][message_id][i] = (emoji, max(0, count + change))
            return

    # If emoji is not found, add it with a count of 1 (only when adding)
    if change > 0:
        GLOBAL_REACTIONS[guild_id][message_id].append((emoji, 1))


@bot.event
async def on_reaction_add(reaction, user):
    """
    Fires when a user adds a reaction to a message.
    Updates the GLOBAL_REACTIONS dictionary.
    """
    guild_id = reaction.message.guild.id
    message_id = reaction.message.id
    emoji = reaction.emoji
    update_reaction(guild_id, message_id, emoji, +1)
    print(f"Reaction added: {emoji} on Message ID: {message_id}")


@bot.event
async def on_reaction_remove(reaction, user):
    """
    Fires when a user removes a reaction from a message.
    Updates the GLOBAL_REACTIONS dictionary.
    """
    guild_id = reaction.message.guild.id
    message_id = reaction.message.id
    emoji = reaction.emoji
    update_reaction(guild_id, message_id, emoji, -1)
    print(f"Reaction removed: {emoji} on Message ID: {message_id}")




def should_combine_with_last_message(guild_id, channel_id, new_message, time_threshold_minutes=5):
    """
    Checks if the new message should be combined with the last message in the channel.
    Conditions:
    - Same user
    - No replies or reactions
    - Sent within the time threshold
    """
    # Check if there are any previous messages to combine with
    if len(GLOBAL_MESSAGES[guild_id][channel_id]) == 0:
        return False

    # Get the last message
    last_message = GLOBAL_MESSAGES[guild_id][channel_id][-1]
    
    # If last_message is a list (already combined), get the first message's data
    if isinstance(last_message, list):
        last_message = last_message[0]

    # Parse timestamps
    last_timestamp = datetime.fromisoformat(last_message[2])
    new_timestamp = datetime.fromisoformat(new_message[2])

    # Check conditions
    same_user = last_message[1] == new_message[1]
    within_time_threshold = (new_timestamp - last_timestamp).total_seconds() / 60 <= time_threshold_minutes

    # Ensure the last message has no replies or reactions
    is_last_message_isolated = (
        last_message[0] not in GLOBAL_REPLY_CHAINS[guild_id] and
        (guild_id not in GLOBAL_REACTIONS or last_message[0] not in GLOBAL_REACTIONS[guild_id])
    )

    return same_user and within_time_threshold and is_last_message_isolated


def combine_with_last_message(guild_id, channel_id, new_message):
    """
    Combines the new message content with the last message in the channel.
    """
    last_message = GLOBAL_MESSAGES[guild_id][channel_id].pop()
    
    # Handle if last_message is already a list of combined messages
    if isinstance(last_message, list):
        last_content = last_message[0][3]
    else:
        last_content = last_message[3]
    
    # Create the combined message
    combined_content = f"{last_content} {new_message[3]}"
    
    # If last_message was a list, use its first message's metadata
    if isinstance(last_message, list):
        combined_message = (last_message[0][0], last_message[0][1], last_message[0][2], combined_content)
    else:
        combined_message = (last_message[0], last_message[1], last_message[2], combined_content)
    
    GLOBAL_MESSAGES[guild_id][channel_id].append(combined_message)



def combine_isolated_message_blocks(guild_id, time_threshold_minutes=5):
    """
    Combines consecutive messages from the same user if they are isolated (no replies, no reactions).
    """
    for channel_id, messages in GLOBAL_MESSAGES[guild_id].items():
        combined_messages = []
        current_block = []
        last_author = None
        last_timestamp = None

        for message_id, author_id, timestamp, content in messages:
            timestamp = datetime.fromisoformat(timestamp)

            # Check if this message is isolated (no replies and no reactions)
            is_isolated = (
                message_id not in GLOBAL_REPLY_CHAINS[guild_id] and
                (guild_id not in GLOBAL_REACTIONS or message_id not in GLOBAL_REACTIONS[guild_id])
            )

            # Check if the current message can be combined with the previous block
            if (last_author == author_id and
                is_isolated and
                last_timestamp and (timestamp - last_timestamp).total_seconds() / 60 <= time_threshold_minutes):
                # Add to the current block
                current_block.append((message_id, author_id, timestamp.isoformat(), content))
            else:
                # Save the current block and start a new one
                if current_block:
                    combined_messages.append(current_block)
                current_block = [(message_id, author_id, timestamp.isoformat(), content)]

            last_author = author_id
            last_timestamp = timestamp

        # Add the last block if it exists
        if current_block:
            combined_messages.append(current_block)

        # Flatten combined messages (single message blocks remain as they are)
        flattened_messages = [
            block if len(block) == 1 else combine_message_block(block) for block in combined_messages
        ]
        GLOBAL_MESSAGES[guild_id][channel_id] = flattened_messages


def combine_message_block(block):
    """
    Combines a block of messages into a single message entry.
    """
    combined_content = " ".join([msg[3] for msg in block])
    first_message = block[0]
    return (first_message[0], first_message[1], first_message[2], combined_content)


def build_conversation_chain(message_id, guild_id):
    """
    Recursively build a conversation chain by following reply references.
    """
    chain = [message_id]
    current_message_id = message_id

    # Traverse upward through replies
    while True:
        found_parent = False
        for parent_id, replies in GLOBAL_REPLY_CHAINS[guild_id].items():
            if current_message_id in replies:
                chain.insert(0, parent_id)  # Prepend the parent to the chain
                current_message_id = parent_id
                found_parent = True
                break
        if not found_parent:
            break  # No parent found, we have reached the root

    return chain


async def gather_reactions_for_guild(guild_id):
    """
    Fetch reactions for each message stored globally for the specified guild
    and populate GLOBAL_REACTIONS with reaction data.
    """
    for channel_id, messages in GLOBAL_MESSAGES[guild_id].items():
        channel = bot.get_channel(channel_id)
        if channel is None:
            continue

        for message_entry in messages:
            if isinstance(message_entry, list):
                message_id, author_id, timestamp, content = message_entry[0]
            else:
                message_id, author_id, timestamp, content = message_entry

            try:
                fetched_message = await channel.fetch_message(message_id)
            except (discord.NotFound, discord.Forbidden):
                continue
            except discord.HTTPException as e:
                print(f"HTTP Exception fetching message {message_id} in #{channel.name}: {e}")
                continue

            if message_id not in GLOBAL_REACTIONS[guild_id]:
                GLOBAL_REACTIONS[guild_id][message_id] = []

            GLOBAL_REACTIONS[guild_id][message_id] = [
                (reaction.emoji, reaction.count) for reaction in fetched_message.reactions
            ]

            await asyncio.sleep(0.1)  # To prevent hitting rate limits


def print_reaction_summary(guild_id):
    """
    Print a summary of the messages and their reactions for the specified guild.
    """
    total_reactions = 0
    for message_id, reactions in GLOBAL_REACTIONS[guild_id].items():
        print(f"Message ID: {message_id}")
        for emoji, count in reactions:
            print(f"  - Reaction: {emoji}, Count: {count}")
            total_reactions += count
    print(f"Total reactions collected for guild {guild_id}: {total_reactions}")


def combine_isolated_message_blocks(guild_id, time_threshold_minutes=5):
    """
    Combines consecutive messages from the same user if they are isolated (no replies, no reactions).
    """
    for channel_id, messages in GLOBAL_MESSAGES[guild_id].items():
        combined_messages = []
        current_block = []
        last_author = None
        last_timestamp = None

        for message_id, author_id, timestamp, content in messages:
            timestamp = datetime.fromisoformat(timestamp)

            # Check if this message is isolated (no replies and no reactions)
            is_isolated = (
                message_id not in GLOBAL_REPLY_CHAINS[guild_id] and
                (guild_id not in GLOBAL_REACTIONS or message_id not in GLOBAL_REACTIONS[guild_id])
            )

            # Check if the current message can be combined with the previous block
            if (last_author == author_id and
                is_isolated and
                last_timestamp and (timestamp - last_timestamp).total_seconds() / 60 <= time_threshold_minutes):
                # Add to the current block
                current_block.append((message_id, author_id, timestamp.isoformat(), content))
            else:
                # Save the current block and start a new one
                if current_block:
                    combined_messages.append(current_block)
                current_block = [(message_id, author_id, timestamp.isoformat(), content)]

            last_author = author_id
            last_timestamp = timestamp

        # Add the last block if it exists
        if current_block:
            combined_messages.append(current_block)

        # Flatten combined messages (single message blocks remain as they are)
        flattened_messages = [
            block if len(block) == 1 else combine_message_block(block) for block in combined_messages
        ]
        GLOBAL_MESSAGES[guild_id][channel_id] = flattened_messages


def calculate_influence_scores():
    """
    Function to calculate influence scores for each user, incorporating reply chains, conversation chains,
    and combined isolated messages.
    """
    guild_scores = {}

    for guild_id, channels in GLOBAL_MESSAGES.items():
        user_message_counts = {}
        user_reactions_received = {}
        user_reply_lengths = {}
        user_chain_scores = {}

        # Step 1: Process messages and gather data
        for channel_id, messages in channels.items():
            for message in messages:
                # If the message is a combined block, use the length as message weight
                if isinstance(message, list):
                    message_weight = len(message)
                    message_id, author_id, timestamp, content = message[0]
                else:
                    message_weight = 1
                    message_id, author_id, timestamp, content = message

                timestamp = datetime.fromisoformat(timestamp)

                # Apply extra weighting for messages with >2 replies
                reply_count = len(GLOBAL_REPLY_CHAINS[guild_id].get(message_id, []))
                reply_weight_multiplier = 1.5 if reply_count > 2 else 1.0
                weighted_message_count = message_weight * reply_weight_multiplier

                # Update message count
                if author_id not in user_message_counts:
                    user_message_counts[author_id] = 0
                user_message_counts[author_id] += weighted_message_count

                # Track reply chain length
                if author_id not in user_reply_lengths:
                    user_reply_lengths[author_id] = 0
                user_reply_lengths[author_id] += reply_count

                # Count reactions
                if guild_id in GLOBAL_REACTIONS and message_id in GLOBAL_REACTIONS[guild_id]:
                    total_reactions = sum(count for emoji, count in GLOBAL_REACTIONS[guild_id][message_id])
                    if author_id not in user_reactions_received:
                        user_reactions_received[author_id] = 0
                    user_reactions_received[author_id] += total_reactions

        # Step 2: Prepare the DataFrame
        data = {
            'user_id': list(set(user_message_counts.keys())),
            'message_count': [user_message_counts.get(user, 0) for user in user_message_counts.keys()],
            'reactions_received': [user_reactions_received.get(user, 0) for user in user_message_counts.keys()],
            'reply_chain_length': [user_reply_lengths.get(user, 0) for user in user_message_counts.keys()]
        }

        df = pd.DataFrame(data)

        # Step 3: Normalize the metrics
        df['normalized_messages'] = df['message_count'] / (df['message_count'].max() or 1)
        df['normalized_reactions'] = df['reactions_received'] / (df['reactions_received'].max() or 1)
        df['normalized_reply_chain'] = df['reply_chain_length'] / (df['reply_chain_length'].max() or 1)

        # Step 4: Calculate the influence score
        df['influence_score'] = (
            0.4 * df['normalized_reactions'] +
            0.3 * df['normalized_reply_chain'] +
            0.3 * df['normalized_messages']
        )

        # Store the result
        guild_scores[guild_id] = df[['user_id', 'influence_score']].sort_values(by='influence_score', ascending=False).reset_index(drop=True)

    return guild_scores

async def classify_message_with_gpt(message_content, parent_message=None, guild_id=None, message_id=None):
    """
    Uses GPT-4 to classify a message as 'cooperative', 'competitive', or 'neutral'.
    Optionally provides context from the parent message if the current message is a reply.
    
    Args:
        message_content: The content of the message to classify.
        parent_message: The content of the parent message if the current message is a reply.
        
    Returns:
        Classification as 'cooperative', 'competitive', or 'neutral'.
    """
    system_prompt = "You are a helpful assistant that classifies messages into three categories: 'cooperative', 'competitive', or 'neutral'. A cooperative message promotes collaboration, help, or positive engagement. A competitive message suggests disagreement, argument, or competitive behavior. A neutral message neither promotes cooperation nor competition."

    user_prompt = f"""
Classify the following message accordingly:

Message: "{message_content}"

{f'Parent Message: "{parent_message}"' if parent_message else ''}

Just return one of: 'cooperative', 'competitive', or 'neutral'.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=10,
            temperature=0
        )
        classification = response.choices[0].message.content.strip().lower()
        if classification == "cooperative" and guild_id is not None and message_id is not None:
            if guild_id not in COLLABORATIVE_MESSAGES:
                COLLABORATIVE_MESSAGES[guild_id] = []
            COLLABORATIVE_MESSAGES[guild_id].append(message_id)
        return classification
    except Exception as e:
        print(f"Error classifying message with GPT-4: {e}")
        return "neutral"


POSITIVE_EMOJIS = [
    '👍', '❤️', '😂', '😊', '🎉', '🙌', '😍', '😁', '😎', '😄', '😆', 
    '💪', '🔥', '👏', '💖', '✨', '🌟', '🍀', '🥳', '😌', '😇', '🤩', 
    '🌈', '😺', '💯', '🤗', '😻', '👌', '🌸', '💐', '✅', '🎁', 
    '🤝', '😃', '🥰', '👊', '😺', '🫶', '🎶', '🕺', '💃', '💎'
]

NEGATIVE_EMOJIS = [
    '👎', '😠', '😢', '💔', '😡', '😞', '😩', '😰', '😭', '🤬', '🤯', 
    '😣', '😖', '💀', '☠️', '😒', '😤', '🤮', '😫', '😕', '😟', 
    '😓', '😔', '😥', '🤢', '😢', '😧', '😾', '🙀', '🤡', '❌', 
    '😬', '🖕', '💣', '👿', '😶‍🌫️', '😤', '👻'
]


async def get_contribution_spirit_scores():
    contribution_scores = {}

    for guild_id, channels in GLOBAL_MESSAGES.items():
        user_contributions = {
            'user_id': [],
            'cooperative_score': [],
            'competitive_score': [],
            'neutral_score': []
        }

        for channel_id, messages in channels.items():
            for message in messages:
                if isinstance(message, list):
                    message_id, author_id, timestamp, content = message[0]
                else:
                    message_id, author_id, timestamp, content = message

                # Get parent message if available (for replies)
                parent_message = None
                if message_id in GLOBAL_REPLY_CHAINS[guild_id]:
                    parent_ids = GLOBAL_REPLY_CHAINS[guild_id][message_id]
                    if parent_ids:
                        parent_message_id = parent_ids[0]
                        parent_message = next(
                            (msg[3] for ch_id, msgs in GLOBAL_MESSAGES[guild_id].items() for msg in msgs if msg[0] == parent_message_id),
                            None
                        )

                # Classify the message using GPT-4
                classification = await classify_message_with_gpt(content, parent_message, guild_id, message_id)

                # Reaction counts
                reactions = GLOBAL_REACTIONS[guild_id].get(message_id, [])
                positive_reaction_count = sum(count for emoji, count in reactions if emoji in POSITIVE_EMOJIS)
                negative_reaction_count = sum(count for emoji, count in reactions if emoji in NEGATIVE_EMOJIS)

                # Determine contribution type and assign scores
                if classification == "cooperative":
                    score = 1.5 + (positive_reaction_count * 0.5)
                    contribution_type = 'cooperative'
                elif classification == "competitive":
                    score = 1.5 + (negative_reaction_count * 0.5)
                    contribution_type = 'competitive'
                else:
                    score = 1  # Neutral base score
                    contribution_type = 'neutral'

                # Update user contribution scores
                if author_id not in user_contributions['user_id']:
                    user_contributions['user_id'].append(author_id)
                    user_contributions['cooperative_score'].append(0)
                    user_contributions['competitive_score'].append(0)
                    user_contributions['neutral_score'].append(0)

                user_index = user_contributions['user_id'].index(author_id)
                if contribution_type == 'cooperative':
                    user_contributions['cooperative_score'][user_index] += score
                elif contribution_type == 'competitive':
                    user_contributions['competitive_score'][user_index] += score
                elif contribution_type == 'neutral':
                    user_contributions['neutral_score'][user_index] += score

        # Normalize and calculate the final score
        df = pd.DataFrame(user_contributions)
        df['normalized_cooperative'] = df['cooperative_score'] / (df['cooperative_score'].max() or 1)
        df['normalized_competitive'] = df['competitive_score'] / (df['competitive_score'].max() or 1)
        df['normalized_neutral'] = df['neutral_score'] / (df['neutral_score'].max() or 1)

        # Calculate overall contribution spirit score
        df['contribution_spirit_score'] = (
            0.5 * df['normalized_cooperative'] - 
            0.3 * df['normalized_competitive'] + 
            0.2 * df['normalized_neutral']
        )

        df['contribution_spirit_score'] = df['contribution_spirit_score'].clip(0, 1)

        # Store the scores for this guild
        contribution_scores[guild_id] = df[['user_id', 'contribution_spirit_score']].sort_values(
            by='contribution_spirit_score', ascending=False
        ).reset_index(drop=True)

    return contribution_scores


async def extract_topics_for_all_guilds():
    """
    Extract topics of discussion for all guilds using GPT-4, 
    with enriched context from reply chains, conversation chains, and reactions.
    
    Returns:
        A dictionary containing topics and their relevance for each guild.
    """
    all_guild_topics = {}

    for guild_id in GLOBAL_MESSAGES.keys():
        messages_data = []

        for channel_id, messages in GLOBAL_MESSAGES[guild_id].items():
            for message in messages:
                if isinstance(message, list):
                    message_id, author_id, timestamp, content = message[0]
                else:
                    message_id, author_id, timestamp, content = message

                # Get relative time in natural language (simplified for demonstration purposes)
                relative_time = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")

                # Fetch parent message if available
                parent_message_content = None
                if message_id in GLOBAL_REPLY_CHAINS[guild_id]:
                    parent_ids = GLOBAL_REPLY_CHAINS[guild_id][message_id]
                    if parent_ids:
                        parent_message_id = parent_ids[0]
                        parent_message_content = next(
                            (msg[3] for ch_id, msgs in GLOBAL_MESSAGES[guild_id].items() for msg in msgs if msg[0] == parent_message_id),
                            None
                        )

                # Summarize the conversation chain if the message is part of one
                conversation_summary = None
                for chain in GLOBAL_CONVERSATION_CHAINS[guild_id]:
                    if message_id in chain:
                        conversation_summary = summarize_conversation_chain(chain, guild_id)
                        break

                # Get reaction counts
                positive_reactions = sum(
                    count for emoji, count in GLOBAL_REACTIONS[guild_id].get(message_id, []) if emoji in POSITIVE_EMOJIS
                )
                negative_reactions = sum(
                    count for emoji, count in GLOBAL_REACTIONS[guild_id].get(message_id, []) if emoji in NEGATIVE_EMOJIS
                )

                # Prepare the message entry
                messages_data.append({
                    "USER": f"User {author_id}",
                    "TIME": relative_time,
                    "MESSAGE": content,
                    "PARENT_MESSAGE": parent_message_content if parent_message_content else "None",
                    "CONVERSATION_CHAIN_SUMMARY": conversation_summary if conversation_summary else "None",
                    "REACTIONS": {
                        "POSITIVE": positive_reactions,
                        "NEGATIVE": negative_reactions
                    }
                })

        messages_object = {"MESSAGES": messages_data}

        # Explain the structure of the data to GPT-4
        explanation_of_format = """
        You are given a JSON object containing a series of messages from a chat. 
        Each message contains the following fields:

        - USER: The user who sent the message.
        - TIME: The time the message was sent (formatted in natural language).
        - MESSAGE: The content of the message.
        - PARENT_MESSAGE: The content of the parent message if this message is a reply.
          If it's not a reply, this field will be "None".
        - CONVERSATION_CHAIN_SUMMARY: A summary of the conversation chain this message is part of.
          If it's not part of any conversation chain, this field will be "None".
        - REACTIONS: An object containing the number of positive and negative reactions to the message.

        Your task is to identify the main topics of discussion within this conversation and return them 
        as a JSON object in the following format:

        {
          "TOPICS": [
            {
              "TOPIC_NAME": "Concise topic name (max 3 words)",
              "TOPIC_RELEVANCE": "Score from 1 to 100 representing how frequently the topic is referenced"
            },
            ...
          ]
        }
        """

        # Create the prompt
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""
        The messages were sent relative to the current datetime: {current_datetime}.
        
        {explanation_of_format}
        
        Here is the JSON object of messages:
        
        {messages_object}
        """

        # Call GPT-4 to generate the topics for the current guild
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that extracts topics from conversations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            topics_json = response.choices[0].message.content.strip().replace("```json", '').replace("```", '')
            print(topics_json)
            all_guild_topics[guild_id] = eval(topics_json)  # Store topics for the current guild

        except Exception as e:
            print(f"Error generating topics for guild {guild_id} with GPT-4: {e}")
            all_guild_topics[guild_id] = {"TOPICS": []}  # Return an empty structure on error

    return all_guild_topics


def summarize_conversation_chain(chain, guild_id):
    """
    Summarizes a conversation chain by extracting key message content or highlights.
    """
    summarized_messages = []
    for message_id in chain:
        message = next(
            (msg for ch_id, msgs in GLOBAL_MESSAGES[guild_id].items() for msg in msgs if msg[0] == message_id),
            None
        )
        if message:
            summarized_messages.append(message[3])  # Extract the message content

    # Concatenate the first few messages or key highlights (limit to 3-4)
    return " -> ".join(summarized_messages[:3]) + ("..." if len(summarized_messages) > 3 else "")


# Example usage:
async def display_all_guild_topics():
    topics = await extract_topics_for_all_guilds()
    print("Extracted Topics for All Guilds:")
    for guild_id, topic_data in topics.items():
        print(f"\nGuild ID: {guild_id}")
        print(topic_data)



def find_general_channel(guild):
    """Find the general channel or return a fallback text channel."""
    general = discord.utils.get(guild.text_channels, name="general")
    if general and general.permissions_for(guild.me).send_messages:
        return general
    # Otherwise, return the first available text channel
    return next(
        (channel for channel in guild.text_channels if channel.permissions_for(guild.me).send_messages), 
        None
    )

class WeeklyStatsView(View):
    def __init__(self, guild_id):
        print(f"Initializing WeeklyStatsView with guild_id: {guild_id}")
        super().__init__()
        self.guild_id = guild_id

    @discord.ui.button(label="View Weekly Summary", style=discord.ButtonStyle.primary)
    async def show_summary(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            print(f"Button clicked for guild {self.guild_id}")
            print(f"Available reports: {list(WEEKLY_REPORTS.keys())}")
            
            if self.guild_id in WEEKLY_REPORTS:
                print(f"Found report for guild {self.guild_id}")
                await interaction.response.send_message(WEEKLY_REPORTS[self.guild_id], ephemeral=True)
                print("Successfully sent report")
            else:
                print(f"No report found for guild {self.guild_id}")
                await interaction.response.send_message("No report available.", ephemeral=True)
        except Exception as e:
            print(f"Error in show_summary: {str(e)}")
            await interaction.response.send_message("An error occurred while retrieving the report.", ephemeral=True)

async def weekly_task():
    print("Running weekly task")
    influence_scores = calculate_influence_scores()
    topics = await extract_topics_for_all_guilds()
    
    # Add backup after weekly processing
    backup_global_dictionaries()
    
    for guild_id, scores in influence_scores.items():
        print("reached loop")
        try:
            guild = bot.get_guild(guild_id)
            print(f"Got guild: {guild}")
            if not guild:
                print(f"Guild not found for ID: {guild_id}")
                continue

            # Find general channel or fallback
            channel = find_general_channel(guild)
            print(f"Found channel: {channel}")
            if not channel:
                print(f"No suitable channel found in guild {guild.name} ({guild_id})")
                continue

            # Debug print to see the structure of scores
            print(f"Scores data: {scores}")

            # Create summary message - modified to handle different data structures
            try:
                if isinstance(scores, list):
                    influence_summary = "\n".join(
                        [f"User {score['user_id']}: {score['influence_score']}" for score in scores]
                    )
                else:
                    # If scores is not in the expected format, create a simpler summary
                    influence_summary = str(scores)
            except Exception as e:
                print(f"Error creating influence summary: {str(e)}")
                influence_summary = "Error processing influence scores"

            try:
                topic_summary = "\n".join(
                    [f"{topic['TOPIC_NAME']} - Relevance: {topic['TOPIC_RELEVANCE']}" for topic in topics[guild_id]["TOPICS"]]
                )
            except Exception as e:
                print(f"Error creating topic summary: {str(e)}")
                topic_summary = "Error processing topics"

            stats_message = f"**Influence Scores:**\n{influence_summary}\n\n**Key Topics:**\n{topic_summary}"
            print("Created stats message")

            # Store the summary in WEEKLY_REPORTS
            WEEKLY_REPORTS[guild_id] = stats_message
            print("Stored in WEEKLY_REPORTS")

            # Send a message with a button to view the weekly summary
            await channel.send("Here is this week's server activity report:", view=WeeklyStatsView(guild_id))
            print("Sent message successfully")

        except Exception as e:
            print(f"Error processing guild {guild_id}: {str(e)}")

def run_scheduled_task():
    asyncio.run_coroutine_threadsafe(weekly_task(), bot.loop)

schedule.every().sunday.at("00:59").do(run_scheduled_task)

async def run_scheduler():
    while True:
        schedule.run_pending()
        await asyncio.sleep(1)  # Add sleep to prevent high CPU usage

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    await bot.change_presence(activity=discord.Game("_scan help"))
    
    # Sync commands with Discord
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Error syncing commands: {e}")
        
    bot.loop.create_task(run_scheduler())

@bot.tree.command(name="serverstats", description="Display detailed server statistics and analytics")
async def server_stats(interaction: discord.Interaction):
    """Display detailed server statistics and analytics"""
    # Defer the response immediately since stats might take time to generate
    await interaction.response.defer()
    
    try:
        guild_id = interaction.guild_id
        
        embed = discord.Embed(
            title=f"📊 Server Statistics for {interaction.guild.name}",
            color=discord.Color.blue(),
            timestamp=datetime.now()
        )
        
        # Get influence scores
        influence_scores = calculate_influence_scores()
        guild_scores = influence_scores.get(guild_id, [])
        
        # Get top 5 influential users
        top_users = ""
        if isinstance(guild_scores, pd.DataFrame):
            top_5 = guild_scores.head(5)
            for _, row in top_5.iterrows():
                user = interaction.guild.get_member(int(row['user_id']))
                username = user.name if user else f"User {row['user_id']}"
                score = round(row['influence_score'] * 100, 1)
                top_users += f"👤 {username}: {score}%\n"
        
        embed.add_field(
            name="🌟 Most Influential Members",
            value=top_users if top_users else "No data available",
            inline=False
        )
        
        # Get server activity metrics
        total_messages = sum(len(messages) for messages in GLOBAL_MESSAGES.get(guild_id, {}).values())
        total_reactions = sum(
            sum(count for _, count in reactions)
            for msg_reactions in GLOBAL_REACTIONS.get(guild_id, {}).values()
            for reactions in [msg_reactions]
        )
        
        # Calculate reply percentage
        reply_count = sum(len(replies) for replies in GLOBAL_REPLY_CHAINS.get(guild_id, {}).values())
        reply_percentage = round((reply_count / total_messages * 100) if total_messages > 0 else 0, 1)
        
        activity_stats = (
            f"📨 Total Messages: {total_messages}\n"
            f"💬 Reply Rate: {reply_percentage}%\n"
            f"😀 Total Reactions: {total_reactions}\n"
            f"🤝 Collaborative Messages: {len(COLLABORATIVE_MESSAGES.get(guild_id, []))}"
        )
        
        embed.add_field(
            name="📈 Activity Metrics",
            value=activity_stats,
            inline=False
        )
        
        # Get topics
        topics = await extract_topics_for_all_guilds()
        guild_topics = topics.get(guild_id, {"TOPICS": []})
        
        top_topics = ""
        for topic in guild_topics["TOPICS"][:3]:  # Top 3 topics
            relevance = topic['TOPIC_RELEVANCE']
            topic_name = topic['TOPIC_NAME']
            top_topics += f"📌 {topic_name}: {relevance}% relevance\n"
        
        embed.add_field(
            name="🗣️ Trending Topics",
            value=top_topics if top_topics else "No topics analyzed yet",
            inline=False
        )
        
        # Add footer with timestamp
        embed.set_footer(text="Stats updated")
        
        # Use followup instead of response.send_message since we deferred
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        print(f"Error generating server stats: {str(e)}")
        # Use followup for error message too
        await interaction.followup.send("An error occurred while generating server statistics.")

@bot.tree.command(name="mystats", description="Display personal user statistics and analytics")
async def my_stats(interaction: discord.Interaction):
    """Display personal user statistics and analytics"""
    try:
        guild_id = interaction.guild_id
        user_id = interaction.user.id
        
        embed = discord.Embed(
            title=f"📊 Personal Statistics for {interaction.user.name}",
            color=discord.Color.green(),
            timestamp=datetime.now()
        )

        # Get message statistics
        total_messages = 0
        for channel_messages in GLOBAL_MESSAGES.get(guild_id, {}).values():
            total_messages += sum(1 for msg in channel_messages if 
                                (isinstance(msg, tuple) and msg[1] == user_id) or
                                (isinstance(msg, list) and msg[0][1] == user_id))

        # Get reaction statistics
        reactions_given = 0
        reactions_received = 0
        for message_reactions in GLOBAL_REACTIONS.get(guild_id, {}).values():
            for emoji, count in message_reactions:
                if emoji in POSITIVE_EMOJIS or emoji in NEGATIVE_EMOJIS:
                    reactions_received += count

        # Calculate reply engagement
        replies_made = 0
        replies_received = 0
        for parent_id, replies in GLOBAL_REPLY_CHAINS.get(guild_id, {}).items():
            if any(msg[1] == user_id for channel in GLOBAL_MESSAGES[guild_id].values() 
                  for msg in channel if isinstance(msg, tuple) and msg[0] == parent_id):
                replies_received += len(replies)
            replies_made += sum(1 for reply_id in replies if 
                              any(msg[1] == user_id for channel in GLOBAL_MESSAGES[guild_id].values() 
                                  for msg in channel if isinstance(msg, tuple) and msg[0] == reply_id))

        # Get influence score
        influence_scores = calculate_influence_scores()
        user_influence = 0
        if guild_id in influence_scores:
            user_row = influence_scores[guild_id][
                influence_scores[guild_id]['user_id'] == user_id
            ]
            if not user_row.empty:
                user_influence = round(float(user_row['influence_score'].iloc[0]) * 100, 1)

        # Get collaborative messages
        collaborative_count = sum(
            1 for msg_id in COLLABORATIVE_MESSAGES.get(guild_id, [])
            if any(msg[0] == msg_id and msg[1] == user_id 
                  for channel in GLOBAL_MESSAGES[guild_id].values() 
                  for msg in channel if isinstance(msg, tuple))
        )

        # Add Activity Stats field
        activity_stats = (
            f"📨 Total Messages: {total_messages}\n"
            f"💬 Replies Made: {replies_made}\n"
            f"📥 Replies Received: {replies_received}\n"
            f"😀 Reactions Received: {reactions_received}\n"
            f"🤝 Collaborative Messages: {collaborative_count}"
        )
        embed.add_field(
            name="📈 Activity Metrics",
            value=activity_stats,
            inline=False
        )

        # Add Influence Score field
        influence_info = (
            f"🌟 Server Influence Score: {user_influence}%\n"
            f"📊 Based on engagement, reactions, and message impact"
        )
        embed.add_field(
            name="🎯 Influence Rating",
            value=influence_info,
            inline=False
        )

        # Add Engagement Style field
        style_metrics = []
        if collaborative_count > 0:
            style_metrics.append("🤝 Collaborative Contributor")
        if replies_made > replies_received:
            style_metrics.append("💭 Active Responder")
        if reactions_received > total_messages/2:
            style_metrics.append("⭐ Impactful Communicator")
        if total_messages > 100:
            style_metrics.append("📢 Regular Participant")

        engagement_style = "\n".join(style_metrics) if style_metrics else "Still building your profile!"
        
        embed.add_field(
            name="🎭 Engagement Style",
            value=engagement_style,
            inline=False
        )

        # Add footer
        embed.set_footer(text="Stats based on server activity | Updated just now")
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        print(f"Error generating user stats: {str(e)}")
        await interaction.response.send_message("An error occurred while generating your statistics.")

if __name__ == "__main__":
    load_global_dictionaries()  # Load saved data
    bot.run(API_TOKEN)