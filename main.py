import random
import pandas as pd
from nltk.tokenize import word_tokenize

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from transaction import getGrocery, findIntent, addToCart, extractQuantityFromInput, extractQuantityFromCart, calculateTotalPrice

intentThreshold = 0.3 #change sensitivity of intent matching
stopPhrase = 'stop'

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stemmed = PorterStemmer()
stopWords = set(stopwords.words('english'))
intentData = pd.read_csv("dialogue.csv")
qaData = pd.read_csv("COMP3074-CW1-Dataset.csv")
groceryData = pd.read_csv("Groceries_dataset_updated.csv")
vectorizer = TfidfVectorizer()

def preprocessText(text, removeStopwords):
    tokens = word_tokenize(text.lower())

    if removeStopwords: #toggle to include stopwords or not
        tokens = [stemmed.stem(token) for token in tokens if token.isalpha() and token not in stopWords]
    else:
        tokens = [stemmed.stem(token) for token in tokens if token.isalpha()]

    return " ".join(tokens)

combinedData = pd.concat([intentData['text'], qaData['Question'], groceryData['itemDescription']]) #create a vocabulary involving all three datasetss
combinedData = combinedData.apply(lambda x: preprocessText(x, False)) #process vocabulary
combinedVectors = vectorizer.fit_transform(combinedData) #assign weight words

#split the vectors according to purpose
intentVectors = combinedVectors[:len(intentData)] 
qaVectors = combinedVectors[len(intentData):len(intentData) + len(qaData)]
groceryVectors = combinedVectors[len(intentData) + len(qaData):]

def getQaAnswer(inputText, vectorizer, qaVectors, qaData):
    inputVector = vectorizer.transform([preprocessText(inputText, False)]) #do not take out stopwords
    similarities = cosine_similarity(inputVector, qaVectors) #work out similarity between input and questions in the qa dataset
    maxSimilarity = similarities.max()
    mostSimilarIntentIndex = similarities.argmax() #find the highest similarity

    if maxSimilarity > intentThreshold:
        mostSimilarIntent = qaData['Answer'].iloc[mostSimilarIntentIndex] #find the most similar value
        return mostSimilarIntent

    return "Sorry, I don't know about this." #qa failed

def addToCart(cart, itemInfo, quantity): #function for adding items to cart
    itemName = itemInfo.split("£")[0].strip()
    for i, cartItem in enumerate(cart): #for every item in the cart, if the item being added is already present
        if itemName in cartItem:
            currentQuantity = extractQuantityFromCart(cartItem) #take the current quantity
            newQuantity = currentQuantity + quantity #add the new quantity
            cart[i] = f"{itemInfo} (Quantity: {newQuantity})"
            return
    cart.append(f"{itemInfo} (Quantity: {quantity})") #item not in cart yet, so add it

def extractQuantityFromCart(cartItem):
    amount = cartItem.split("(Quantity:")[1].split(")")[0].strip() #take quantity value out of cart
    return int(amount) if amount.isdigit() else 1 #check if quantity extracted is a number

def calculateTotalPrice(cart): #sums up the price of all the items in the cart
    totalPrice = sum([float(item.split("£")[1].split()[0]) * extractQuantityFromCart(item) for item in cart])
    return totalPrice

def editCart(cart): #allow user to change the cart
    if not cart: #if there is nothing in the cart, then inform the user and do nothing
        print("Bot: Your shopping cart is empty.")
        return

    print("Bot: Your current shopping cart:")
    for i, item in enumerate(cart, start=1):
        print(f"{i}. {item}") #output item

    while True:
        try: #validation to make sure that user enters numbers, not str
            itemIndex = int(input("Bot: Enter the number of the item you want to edit (0 to cancel): "))
            if 0 <= itemIndex <= len(cart):
                break
            else:
                print("Bot: Please enter a valid item number.")
        except ValueError:
            print("Bot: Please enter a valid item number.")

    if itemIndex == 0:
        print("Bot: Edit cancelled.") #no edit made
        return

    selectedItem = cart[(itemIndex - 1)]

    while True: #loop for input validation. user must enter a number at least 0
        try:
            newQuantity = int(input(f"Bot: Enter the new quantity for {selectedItem} (0 to remove): "))
            if newQuantity >= 0:
                break
            else:
                print("Bot: Please enter a non-negative quantity.")
        except ValueError:
            print("Bot: Please enter a valid number.")

    if newQuantity == 0:
        cart.remove(selectedItem)
        print(f"Bot: Got it. {selectedItem} removed from the cart.")
    else:
        cart[itemIndex - 1] = f"{selectedItem.split('(Quantity:')[0]} (Quantity: {newQuantity})"
        print(f"Bot: Understood! Quantity updated.")

#main conversation begins
nameGiven = False #name is not given yet, default to guest
shoppingCart = []  #initialise shopping cart
print("Welcome to the Grocery Chatbot.")
print("Enter STOP to end the conversation.")
print("Feel free to ask what I can do!")
while True: #while bot is not stopped
    userInput = input("User: ")
    if userInput.lower() == stopPhrase:
        break #stop bot
    else:
        userIntent = findIntent(userInput, vectorizer, intentVectors, intentData) #perform intent matching on user input
        #print(userIntent) (DEBUGGING)
        if userIntent == 'qa':
            print("Bot: Let me look that up for you:")
            answer = getQaAnswer(userInput, vectorizer, qaVectors, qaData)
            print("Bot:", answer)
        else:
            if userIntent == 'greeting':
                print("Bot:", random.choice(["Hi!", "Hello!", "Hey!", "Howdy."]))
            elif userIntent == 'transaction': #detect that user want to buy something
                print("Bot: Okay, let's buy something!") #acknowledge that order process is starting
                transaction = getGrocery(userInput, vectorizer, groceryVectors, groceryData, shoppingCart) #begin transaction
                print(transaction)

                if "Added" in transaction:
                    itemInfo = transaction.split("\n")[0].strip()
                    quantity = extractQuantityFromInput(userInput) #take in which quantity specified by user
                    addToCart(shoppingCart, itemInfo, quantity) #add the specified quantity of items to the cart
            elif userIntent == 'wellbeing':
                print("Bot:", random.choice(["I'm doing well! I hope you are too!", "Doing pretty good, thank you!", "I'm all ready to get you some groceries!"]))
            elif userIntent == 'functions':
                print("Bot: I'm a bot that helps you order groceries. Please tell me to buy something, and I'll start the ordering process. Ask me 'what do you have in stock?' to see examples of what you can buy!\nBot: You can view and edit your shopping cart as well. When you're ready to place your delivery, please request to checkout.\nBot: I can also try to answer your questions ('what is...', 'tell me about...') or just have a simple chat, like greetings, and taking in your name!")
            elif userIntent == 'changeName' and nameGiven: #name has been removed from bot's memory
                print("Bot: Okay, I've forgotten the name you told me.")
                nameGiven = False 
            elif userIntent == 'repeatName':
                while not nameGiven:
                    print("Bot: What's your name?")
                    userName = input("User: ").title()
                    print("Bot: Is", userName, "correct?")
                    confirmation = input("User: ").lower()
                    if confirmation == "yes" or confirmation == "y":
                        print("Bot: Okay. If you would like to change your name, please tell me to forget.")
                        nameGiven = True #bot now remembers name
                    else:
                        print("Bot: Is that so? In that case, I'd like you to tell me again.") #try again
                print(f"Bot: Hi there, {userName}!")
            elif userIntent == 'viewcart':
                if nameGiven:
                    print(f"Bot: Now showing {userName}'s cart...") #add personalisation if user has given their name to the bot
                else:
                    print(f"Bot: Now showing guest's cart...")
                if not shoppingCart:
                    print("Bot: Your shopping cart is empty. Please let me know if you want to order something.")
                else:
                    print("Bot: Your shopping cart:")
                    for item in shoppingCart:
                        print(item)
                    totalPrice = calculateTotalPrice(shoppingCart)
                    print(f"Bot: Your total comes to £{totalPrice:.2f}.")
            elif userIntent == 'checkout':
                if not shoppingCart:
                    print("Bot: Your shopping cart is empty, so there is nothing to checkout. Please let me know if you want to order something.")
                else:
                    totalPrice = calculateTotalPrice(shoppingCart)
                    print(f"Bot: Your total comes to £{totalPrice:.2f}.")
                    confirmation = input("Bot: Do you want to proceed with the checkout? (yes/no): ").lower()

                    if confirmation == 'yes' or confirmation == 'y':
                        if nameGiven:
                            print(f"Bot: Thank you, {userName}! Your items will be delivered.") #add user's name for personalisation
                        else:
                            print("Bot: Thank you! Your items will be delivered.") #user's name not given
                        shoppingCart = []  #empty the shopping cart
                    else:
                        print("Bot: Checkout canceled. No changes have been made to your cart.") #checkout unsuccessful

            elif userIntent == 'viewstock':
                randomItemsData = groceryData.sample(3).drop_duplicates(subset='itemDescription') #get 3 random items as examples of what the user can buy. Do not repeat any items.
                
                response = "Bot: We have many kinds of items, like"
                for index, row in randomItemsData.iterrows(): #go through each row of the random items chosen.
                    item = row['itemDescription']
                    price = float(row['Price']) #show price of each item
                    response += f" {item} (£{price:.2f} each)," #output each of the three random items
                response = response[:-1] #remove the last comma
                response += "!\nBot: Try placing an order for a specific item and we'll let you know if it's available for purchase."

                print(response)
            elif userIntent == 'editcart':
                editCart(shoppingCart)
            else:
                print("Bot: Sorry, I didn't understand.") #no intent detected

print("Bot: It was nice talking to you!") #acknowledge end of conversation
