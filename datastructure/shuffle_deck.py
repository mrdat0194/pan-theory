import random
from collections import namedtuple

PlayingCard = namedtuple('card', ['suit', 'rank'])

suits = ['Spades', 'Diamonds', 'Hearts', 'Clubs']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

class PlayingDeck:
    def __init__(self):
        self.cards = [PlayingCard(rank, suit) for suit in suits for rank in ranks]

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, position):
        return self.cards[position]

    def __setitem__(self, position, card):
        self.cards[position] = card

if __name__ == '__main__':
    deck = PlayingDeck()

    # # 1. Iterating through a deck
    # for card in deck:
    #     print(card)
    #
    # #2. Cutting the deck
    # first_cut = deck[:4]
    # second_cut = deck[4:]
    #
    # #3. Iterating through the deck in reverse
    # for card in reversed(deck):
    #     print(card)

    #4. Shuffling the deck
    random.shuffle(deck)
    for card in deck:
        print(card)
