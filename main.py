# suppress warnings
import warnings

warnings.filterwarnings("ignore")

# import libraries
import requests, os
import argparse
from PIL import Image


import gradio as gr
from together import Together
import textwrap


## FUNCTION 1: This Allows Us to Prompt the AI MODEL
# -------------------------------------------------
def prompt_llm(prompt, with_linebreak=False):
    # This function allows us to prompt an LLM via the Together API

    # model
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

    # Calculate the number of tokens
    tokens = len(prompt.split())

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


## FUNCTION 2: This Allows Us to Generate Images
# -------------------------------------------------
def gen_image(prompt, width=256, height=256):
    # This function allows us to generate images from a prompt
    response = client.images.generate(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell-Free",  # Using a supported model
        steps=2,
        n=1,
    )
    image_url = response.data[0].url
    image_filename = "image.png"

    # Download the image using requests instead of wget
    response = requests.get(image_url)
    with open(image_filename, "wb") as f:
        f.write(response.content)
    img = Image.open(image_filename)
    img = img.resize((height, width))

    return img


## Function 3: This Allows Us to Create a Chatbot
# -------------------------------------------------
def bot_response_function(user_message, chat_history):
    # 1. YOUR CODE HERE - Add your external knowledge here
    external_knowledge = """
    Dungeons and Dragons is a role-playing game where players create characters and explore a fantasy world.
    The game is played with a group of players, each with a character.
    The characters are controlled by the players, and the players are responsible for the actions of their characters.
    The players are also responsible for the story of the game.
    The players are also responsible for the world of the game.
    The players are also responsible for the rules of the game.
    The players are also responsible for the lore of the game.
    """

    # 2. YOUR CODE HERE -  Give the LLM a prompt to respond to the user
    chatbot_prompt = f"""
    You are a Dungeon Master for a Dungeons and Dragons game. You are responsible for the story of the game, the world of the game, and the rules of the game. Remember to be as detailed as possible and use the following information as additional context for your responses:
    
    {external_knowledge}

    Remember to:
        - Build the character based on the user's message
        - And build the world based on the character created
    
    Now, please respond to this message as a Dungeon Master:
    {user_message}
    """

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        messages=[{"role": "user", "content": chatbot_prompt}],
    )
    response = response.choices[0].message.content

    # 3. YOUR CODE HERE - Generate image based on the response
    image_prompt = f"A {response} in a anime art style"
    image = gen_image(image_prompt)

    # Append the response and image to the chat history
    chat_history.append((user_message, response))
    return "", chat_history, image


if __name__ == "__main__":
    # args on which to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=int, default=1)
    parser.add_argument("-k", "--api_key", type=str, default=None)
    args = parser.parse_args()

    # Get Client for your LLMs
    client = Together(api_key=args.api_key)

    # run the script
    if args.option == 1:
        ### Task 1: YOUR CODE HERE - Write a prompt for the LLM to respond to the user
        prompt = "write a 3 line post about pizza"

        # Get Response
        response = prompt_llm(prompt)

        print("\nResponse:\n")
        print(response)
        print("-" * 100)

    elif args.option == 2:
        ### Task 2: YOUR CODE HERE - Write a prompt for the LLM to generate an image
        prompt = "Create an image of a cat"

        print(f"\nCreating Image for your prompt: {prompt} ")
        img = gen_image(prompt=prompt, width=256, height=256)
        os.makedirs("results", exist_ok=True)
        img.save("results/image_option_2.png")
        print("\nImage saved to results/image_option_2.png\n")

    elif args.option == 3:
        ### Task 3: YOUR CODE HERE - Write a prompt for the LLM to generate text and an image
        text_prompt = "write a 3 line post about resident evil for instagram"
        image_prompt = f"give me an image that represents this '{text_prompt}'"

        # Generate Text
        response = prompt_llm(text_prompt, with_linebreak=True)

        print("\nResponse:\n")
        print(response)
        print("-" * 100)

        # Generate Image
        print(f"\nCreating Image for your prompt: {image_prompt}... ")
        img = gen_image(prompt=image_prompt, width=256, height=256)
        img.save("results/image_option_3.png")
        print("\nImage saved to results/image_option_3.png\n")

    elif args.option == 4:
        # 4. Task 4: Create the chatbot interface (see bot_response_function for more details)
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown("## 🤖 AI Chatbot")
            gr.Markdown("Enter your message below and let the chatbot respond!")

            chatbot = gr.Chatbot()
            image_output = gr.Image(label="Generated Image")
            user_input = gr.Textbox(
                placeholder="Type your message here...", label="Your Message"
            )
            send_button = gr.Button("Send")

            send_button.click(
                bot_response_function,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot, image_output],
            )
            user_input.submit(
                bot_response_function,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot, image_output],
            )

        app.launch()
    else:
        print("Invalid option")