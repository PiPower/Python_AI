import  pygame
import  random
import  copy
import  math
import numpy as np
pygame.font.init()
WIN_HEIGHT = 800
WIN_WIDTH = 1200
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
STAT_FONT = pygame.font.SysFont('comicsans', 50)

TopGapAdjustment = 40


class Player:
    def __init__(self,x,y,width,height):
        self.x=x
        self.y=y
        self.epsilon = 0.1
        self.width = width
        self.height = height
        self.Dead = False
        self.Speed = 30
        self.States = np.zeros((800,WIN_HEIGHT*2+40))  #(Estimation)  State is represented by vector in y axis between Top left of block to  Top Left to gap
        #self.TotalReward =  np.zeros(WIN_HEIGHT*2+40)
        #self.Count =  np.zeros(WIN_HEIGHT*2 + 40)
        self.Discount = 0.8

    def Draw( self,Surface ):
         pygame.draw.rect(Surface, (122, 122, 122), (self.x, self.y, self.width, self.height))
    def MoveUp( self ):
        self.y -= self.Speed

    def MoveDown( self ):
        self.y += self.Speed

    def Decision( self,pipe,speedX,epsilon=True ):
        GapYCord = pipe.GapTop + TopGapAdjustment
        n = random.random()

        #x1 = (abs(self.x - pipe.rectTop.x- pipe.rectTop.width-speedX),self.y+self.Speed - GapYCord )
       # x2 = (abs(self.x - pipe.rectTop.x - pipe.rectTop.width - speedX), self.y - self.Speed -GapYCord)
       # x3 = (abs(self.x - pipe.rectTop.x - pipe.rectTop.width - speedX), self.y - GapYCord)

        MoveValues = [
        self.States[abs(self.x - pipe.rectTop.x- pipe.rectTop.width-speedX)][self.y+self.Speed - GapYCord ],
        self.States[abs(self.x - pipe.rectTop.x - pipe.rectTop.width - speedX)] [ self.y - self.Speed - GapYCord ],
        self.States[abs(self.x - pipe.rectTop.x - pipe.rectTop.width - speedX)] [ self.y - GapYCord ]
        ]

        Indexes =[]
        if epsilon and n <= self.epsilon and max(MoveValues) != min(MoveValues) :
            for i in range(3):
                if MoveValues [ i ] != max(MoveValues):
                    Indexes.append(i)
            return random.choice(Indexes), GapYCord
        else:
            for i in range(3):
                if MoveValues[i] == max(MoveValues):
                    Indexes.append( i)

            return  random.choice(Indexes),GapYCord
        ''' if n<=0.3:
                return 0,GapYCord
             if n <= 0.65:
                return 1,GapYCord
             return 2,GapYCord'''


    def LearnMC( self,History):
        Gt =0
        for board in History:
            for DistanceX,DistanceY, Reward in reversed(board):
                Gt = Reward + 0.9*Gt
               # self.Count[DistanceY] += 1
                Ds= self.States[DistanceX][DistanceY] + self.Discount*(Gt - self.States[DistanceX][DistanceY])
                lol = self.States[DistanceX][DistanceY]
                self.States[DistanceX][DistanceY] =  Ds
                lol= self.States[DistanceX][DistanceY]

    def LearnTd( self, History ):
        gamma = 0.8
        if len(History)==1:
            self.States [ History [ 0 ] [ 0 ] ] [ History [ 0 ] [ 1 ] ] +=    History [ 0 ] [ 2]
        else:
            self.States [ History[0][0] ] [ History[0][1] ] =  self.States [ History[0][0] ] [ History[0][1] ] + \
                self.Discount*(History[0][2] + gamma * self.States [ History[1][0] ] [ History[1][1] ]- self.States [ History[0][0] ] [ History[0][1] ] )

class Obsticle:
    Gap = 150

    def __init__(self,x,widht):
        self.x=x
        self.width = widht
        self.GapTop = random.randint(200,WIN_HEIGHT-200)
        self.rectTop = pygame.Rect(x,5,widht,self.GapTop)
        self.rectBottom = pygame.Rect(x, self.GapTop+Obsticle.Gap, widht, WIN_HEIGHT)
        self.passed = False
        self.Finished = False
        self.Marked = False

    def Move( self,deltaX ):
        self.x += deltaX
        self.rectTop.x += deltaX
        self.rectBottom.x += deltaX

    def Collision( self, player ):
        if  not (self.Finished) and  player.x+player.width > self.rectTop.x and (player.y < self.rectTop.y +self.rectTop.height  or player.y+player.height > self.rectBottom.y):
            player.Dead=True
            return True

    def Passed( self,player ):
        if player.x >= self.rectTop.x+ self.rectTop.width:
            self.Finished = True

        if player.x + player.width> self.rectTop.x:
            self.passed=True
            return True
        return False

    def Draw( self,Surface ):
        pygame.draw.rect(Surface,(70, 100, 157),  self.rectTop)
        pygame.draw.rect(Surface, (70, 100, 157), self.rectBottom)

class State:
    def __init__(self,distance,size,Speed=-5):
        self.distance = distance
        self.Speed = Speed
        self.size=size
        self.Obsticles= [Obsticle(distance+100,size) ]

        totalDistance = self.Obsticles[0].rectTop.x + self.Obsticles[0].rectTop.width +distance  +size
        while  totalDistance< WIN_WIDTH:
            self.Obsticles.append( Obsticle(totalDistance , size) )
            totalDistance += distance+size

    def Move( self,player ):
        NextState = copy.deepcopy(self)
        for obs in NextState.Obsticles:
            obs.Move(self.Speed)
            if obs.x + obs.width < 5:
                NextState.Obsticles.remove(obs)

        RectRef=NextState.Obsticles[len(NextState.Obsticles)-1]
        if abs( RectRef.x+RectRef.width - WIN_WIDTH) > NextState.distance:
            NextState.Obsticles.append(Obsticle(WIN_WIDTH,NextState.size))

        return  NextState

    def MoveTraining( self, player ):
        NextState = copy.deepcopy(self)
        for obs in NextState.Obsticles:
            obs.Move(self.Speed)
            if obs.x + obs.width < 5:
                NextState.Obsticles.remove(obs)
        return NextState

    def Collision(self,player):
        score = 0
        for obs in self.Obsticles:
            if not(obs.passed) and obs.Passed(player):
                score = 1
            obs.Passed(player)

            if obs.Collision(player):
                return  True,score

        return False, score


    def Draw( self,Surface ):
        for obs in   self.Obsticles:
            obs.Draw(Surface)

MaxTreeLenght = 3
def Deccision(StateBase,playerBase,TreeSize=0):
    # Going down = 0
    # Going up = 1
    # Not moving = 2

    Rewards = [0,0,0]
    Distances =[0,0,0]
    for i in range(3):
        LocalRewards = [ 0, 0, 0 ]
        State = copy.deepcopy(StateBase)
        player = copy.deepcopy(playerBase)

        if i==0:
            player.MoveDown()
        elif i == 1:
            player.MoveUp()

        NextState = State.MoveTraining(player)
        Coll, result = NextState.Collision(player)

        TopRect = NextState.Obsticles[0].rectTop
        BotRect = NextState.Obsticles[0].rectBottom

        if playerBase.y < TopRect.y+TopRect.height-40 or playerBase.y + playerBase.height > BotRect.y+40 :
            Distances[i]=  math.sqrt((player.x-TopRect.x-TopRect.width)**2 + (player.y-TopRect.y-TopRect.height+ 40)**2)

        if not(Coll) and TreeSize < MaxTreeLenght:
            LocalRewards,_= Deccision(NextState,player,TreeSize+1)

        if result == 1:
            Rewards[i] = 10

        if Coll:
            Rewards[i] = -10
        Rewards[i] += max(LocalRewards)

    for i in range(3):
        if Distances[i] == min(Distances) :
            Rewards[i]+=5

    if  min(Rewards) == max(Rewards):
        n = random.randint(0,2)
        return Rewards, n

    if Rewards[0] == max(Rewards):
        return Rewards, 0
    if Rewards [ 1 ] == max(Rewards):
        return Rewards, 1
    if Rewards [ 2 ] == max(Rewards):
        return Rewards, 2


def Draw(window,player=None,State=None,score = None):
    pygame.draw.rect(window,(255,255,255),(5,5,WIN_WIDTH-5,WIN_HEIGHT-5))

    if player is not None:
        player.Draw(window)
    if State is not None:
        State.Draw(Surface= window)
    if score is not None:
        text = STAT_FONT.render("Score: " + str(score), 1, (0, 0, 0))
        window.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    pygame.display.update()


def TrainWithoutDrawin(player,epochs):
    for i in range(epochs):
        run = True
        player.y = np.random.randint(200, WIN_HEIGHT - 300)
        score = 0
        board = State(400, 50, -20)
        iteration =0
        History = [[]]
        HistoryTD = []
        Learning  = False
        while run:

            # _,choice = Deccision(board,player)
            YCord = None
            for pipe in board.Obsticles:
                if not (pipe.Finished):
                    choice, YCord = player.Decision(pipe,board.Speed)
                    break

            if choice == 0:
                player.MoveDown()
            elif choice == 1:
                player.MoveUp()


            board = board.Move(player)
            Coll, result = board.Collision(player)
            if player.y <= 0 or player.y + player.height >= WIN_HEIGHT:
                Coll = True


            score += result
            run = not (Coll)

            Reward = 0
            if Coll:
                Reward -= 25
            elif result:
                Reward += 25

            for pipe in board.Obsticles:
                if not (pipe.Finished):
                    if Reward == 0:
                        Reward = -0.04 * abs(player.y - pipe.GapTop -TopGapAdjustment) + 10
                        if not(Reward > -15):
                            Reward = -15

                    HistoryTD.append(
                        [ abs(player.x - pipe.rectTop.x - pipe.rectTop.width), player.y - pipe.GapTop-TopGapAdjustment,  Reward ])
                    # History[iteration].append([abs(player.x + player.width  - pipe.rectTop.x- pipe.rectTop.width),player.y - pipe.GapTop,Reward])
                    break
                elif pipe.Finished and not (pipe.Marked):
                    History.append([ ])
                    iteration += 1
                    pipe.Marked = True

            if Learning:
                player.LearnTd(HistoryTD)
                HistoryTD.pop(0)
            if not (run):
                player.LearnTd(HistoryTD)
            Learning = True

        #player.LearnMC(History)
        if i % 100 ==0:
            print("Epoch: "+str(i))


def main():

    pygame.key.set_repeat(1)
    clock = pygame.time.Clock()
    player = Player(90, np.random.randint(200, WIN_HEIGHT - 300), 50, 50)
    TopScore = 0
    TrainWithoutDrawin(player,epochs=6000)
    while True:
        run = True
        player.y = np.random.randint(200, WIN_HEIGHT - 300)
        score = 0
        board = State(400, 50, -20)

        History = [[]]
        HistoryTD = []
        iteration = 0
        Learning  = False
        while run:
        #------ Input
           clock.tick(45)
           for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    player.MoveDown()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    player.MoveUp()
        #------- AI

           #_,choice = Deccision(board,player)
           YCord = None
           for pipe in board.Obsticles:
               if not(pipe.Finished):
                  choice,YCord = player.Decision(pipe,board.Speed,False)
                  break


           if choice == 0:
                player.MoveDown()
           elif choice == 1:
                player.MoveUp()

        #------- Drawing

           board = board.Move(player)
           Coll, result = board.Collision(player)


           score += result
           run = not(Coll)
           Draw(WIN,player,board,score)
           if player.y<=0 or player.y+player.height >= WIN_HEIGHT:
               Coll= True

           Reward = 0
           if Coll:
              Reward -= 25
           elif result:
               Reward += 25


           for pipe in board.Obsticles:
              if not (pipe.Finished):
                if Reward == 0:
                    Reward = -0.04*abs(player.y - pipe.GapTop-TopGapAdjustment) + 10
                    if not (Reward > -15):
                        Reward = -15
                HistoryTD.append([abs(player.x - pipe.rectTop.x- pipe.rectTop.width),player.y - pipe.GapTop-TopGapAdjustment,Reward])
               # History[iteration].append([abs(player.x + player.width  - pipe.rectTop.x- pipe.rectTop.width),player.y - pipe.GapTop,Reward])
                break
              elif pipe.Finished and not (pipe.Marked):
                History.append([ ])
                iteration += 1
                pipe.Marked = True

           if score > TopScore:
                TopScore = score

           if Learning:
               player.LearnTd(HistoryTD)
               HistoryTD.pop(0)
           if not(run):
               player.LearnTd(HistoryTD)
           Learning = True
        #player.LearnMC(History)
main()