from datastructure.story_teller import Player

if __name__ == "__main__":
    # Example data from the docstring in story_teller.py
    data = [
        Player("amy", 100),
        Player("david", 100),
        Player("heraldo", 50),
        Player("aakansha", 75),
        Player("aleksa", 150)
    ]
    
    print("Running score_billboard:")
    Player.score_billboard(data)
