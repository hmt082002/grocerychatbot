import random
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

intentThreshold = 0.3
stemmed = PorterStemmer() #make tokens and turn the words into stems
stopWords = set(stopwords.words('english'))

def preprocessText(text, removeStopwords):
    tokens = word_tokenize(text.lower())
    if removeStopwords:
        tokens = [stemmed.stem(token) for token in tokens if token.isalpha() and token not in stopWords]
    else:
        tokens = [stemmed.stem(token) for token in tokens if token.isalpha()]
    return " ".join(tokens)

def findIntent(inputText, vectorizer, intentVectors, intentData):  #detecting the intent from the user's input
    inputVector = vectorizer.transform([preprocessText(inputText, False)])
    similarities = cosine_similarity(inputVector, intentVectors)  #find the likeness between the input and the intents
    maxSimilarity = similarities.max()
    mostSimilarIntentIndex = similarities.argmax()  #get the highest similarity
    if maxSimilarity > intentThreshold:
        mostSimilarIntent = intentData['intent'].iloc[mostSimilarIntentIndex]  #find the most appropriate intent
        return mostSimilarIntent
    return "null"  #no intent found

def addToCart(cart, itemInfo, quantity): #once confirmed, add item to cart
    itemName = itemInfo.split("£")[0].strip() #get just the item name
    for i, cartItem in enumerate(cart):
        if itemName in cartItem:
            currentQuantity = extractQuantityFromCart(cartItem)
            newQuantity = currentQuantity + quantity
            cart[i] = f"{itemInfo} (Quantity: {newQuantity})"
            return
    cart.append(f"{itemInfo} (Quantity: {quantity})")

def extractQuantityFromCart(cartItem):
    quantity = cartItem.split("(Quantity:")[1].split(")")[0].strip()
    return int(quantity) if quantity.isdigit() else 1

def calculateTotalPrice(cart):
    totalPrice = sum([float(item.split("£")[1].split()[0]) * extractQuantityFromCart(item) for item in cart])
    return totalPrice

def confirmPurchase(item, quantity, price, recommendation):
    price = float(price)
    totalPrice = quantity * price #find total price
    confirmation = input(
        f"Bot: You want to buy {quantity} {'items' if quantity > 1 else 'item'} of {item} "
        f"at £{price:.2f} each. This will be £{totalPrice:.2f} in total. Is that correct? (yes/no): ").lower()

    if confirmation == 'yes' or confirmation == 'y':
        totalPrice = quantity * price
        return f"Bot: Thank you for your order! The total price is £{totalPrice:.2f}. Your items have been added to the cart.\n" \
               f"Bot: Based on what other customers have purchased with {item}, you might also like {recommendation}."
    else:
        return "Bot: Transaction canceled."

def getRecommendation(currentItem, groceryData):
    relatedItems = [] #initialise recommendations
    memberNumbers = groceryData[groceryData['itemDescription'] == currentItem]['Member_number']
    relatedItems = groceryData[groceryData['Member_number'].isin(memberNumbers)]['itemDescription']
    relatedItems = relatedItems[relatedItems != currentItem]
    itemCounts = relatedItems.value_counts() #count the frequencies of related items that a customer buying this item has also bought
    nextMostCommonItem = itemCounts.index[random.randint(0, 2)] #find the top 3 most bought items
    return nextMostCommonItem

def askForQuantity(quantity=None):
    while True: #loop for validation. user must enter a number
        if quantity is not None:
            return quantity
        else:
            quantityInput = input("Bot: How many items do you want to buy? (Specify a number): ")
            try:
                quantity = int(quantityInput)
            except ValueError:
                print("Bot: Please enter a valid number.") #invalid input

def extractQuantityFromInput(inputText): #find numbers in purchase request
    for word in inputText.split():
        if word.isdigit():
            return int(word) #quantity found
    return None #user did not specify quantity

def getGrocery(inputText, vectorizer, groceryVectors, groceryData, shoppingCart): #get the most approriate grocery item based on user input
    inputVector = vectorizer.transform([preprocessText(inputText, True)]) #remove stopwords
    similarities = cosine_similarity(inputVector, groceryVectors)
    similarItemsIndices = similarities.argsort()[0][::-1]

    quantity = extractQuantityFromInput(inputText)
    uniqueItems = set()
    clarificationAttempts = 0

    for index in similarItemsIndices:
        itemCandidate = groceryData['itemDescription'].iloc[index]
        price = groceryData['Price'].iloc[index]

        if itemCandidate not in uniqueItems:
            uniqueItems.add(itemCandidate)
            price = float(price)
            clarification = input(
                f"Bot: Did you mean {itemCandidate}? (£{price:.2f} each) (yes/no/cancel): ").lower()

            clarificationAttempts += 1

            if clarification == 'yes' or clarification == 'y':
                if quantity is None:
                    quantity = askForQuantity()
                recommendation = getRecommendation(itemCandidate, groceryData)
                info = f"{itemCandidate} £{price:.2f}"
                addToCart(shoppingCart, info, quantity)
                return confirmPurchase(itemCandidate, quantity, price, recommendation)

            elif clarification == 'cancel':
                return "Transaction canceled."

            if clarificationAttempts >= 10:
                return "Bot: Sorry that I couldn't find what you were looking for. Transaction canceled."

    return "Bot: Sorry, we don't have that in stock." #no appropriate grocery item found
