
# coding: utf-8

# In[37]:


'''
This block MUST be executed ONCE before running the code. This denotes a global variable since the 
randgen function used is not reproducible and hence a global variable is used to reproduce the results. 
'''

rndness=[]


# In[43]:


#Importing Dependencies
import turtle
import numpy
import random

'''
This function draws a box starting from (x,y) of length l_x and breadth l_y and fills it with specified color.
'''
def draw_box(t,x,y,l_x,l_y,color): #t : board ,x,y coordinates , l_x,l_y side lengths, color to be filled
    if (l_x==0 or l_y==0):
        return 
    t.color(color)
    t.ht()
    t.penup()
    t.goto(x,y)
    t.pendown()
    t.fillcolor(color)
    t.begin_fill()
    t.forward(l_x)
    t.left(90)
    t.forward(l_y)
    t.left(90)
    t.forward(l_x)
    t.left(90)
    t.forward(l_y)
    t.left(90)
    t.end_fill()

'''
This function generates N natural numbers among the given choices, with replacement. For example,
1 is chosen with probability 1/6 and 2 is chosen with probability 1/15
Since the SystemRandom function is used, there is no point of putting a seed since it is anyway ignored.
The produced sequence is NOT reproducible, but it produces random numbers far better than the random.Random object
since it uses the system uncertainities for generating random numbers
'''
def randgen(N,seed=0):
    gen = random.SystemRandom(seed)
    li=[gen.choice([1,1,1,1,1,2,2,3,3,3,3,3,4,4,5,5,6,6,7,7,7,7,7,8,8,9,9,9,9,9]) for i in range(N)]
    return li

'''
Draws a sub-block with 4 different colored rectangles. r_x*l and r_y*b are fractions of length and breadth
respectively.The 2 non-adjacent sub-parts are colored with the same colour. 
r_x and r_y lie between 0 and 1
'''
def draw_sublock(board,r_x,r_y,s_x,s_y,l,b,col):
    x=s_x
    y=s_y
    l_x=r_x*l
    l_y=r_y*b
    for i in range(0,2):
        for j in range(0,2):
            draw_box(board,x,y,l_x,l_y,col[(i+j)%2])
            x+=l_x
            l_x=l-l_x
        x=s_x
        y+=l_y
        l_y=b-l_y

def sublock(board,s_x,s_y,l,b,N_x,N_y,col):
    draw_sublock(board,N_x,N_y,s_x,s_y,l,b,col)

'''
Combines N number of sub-blocks and creates a block with recurrent patterns
'''
def block(board,s_x,s_y,l,b,N,N_x,N_y,col) :
    x=s_x
    y=s_y
    board.hideturtle()
    for i in range (0,N):
        for j in range(0,N):
            sublock(board,x,y,l/N,b/N,N_x,N_y,col)
            x+=l/N
        x=s_x
        y+=b/N
    #print("**Block made**")


'''
makes a superblock which consists of 4 blocks(arranged in a 2X2 fashion). Non adjacent blocks are of the 
same type as well as color. Adjacent blocks are of same type but different colors.
'''
def superblock(board,s_x,s_y,l1,b1,N,N_x,N_y,col1,col2) :
    x=s_x
    y=s_y
    for i in range (0,2):
        for j in range(0,2):
            block(board,x,y,l1/2,b1/2,N,N_x,N_y,[col1[(i+j)%len(col1)], col2[(i+j)%len(col2)]],)
            x+=l1/2
        x=s_x
        y+=b1/2

        
'''
Gets the types of suprblocks for the grid. Since there are n2 rows and n1 columns of superblocks, there are 
n1Xn2 ids which consist of a tuple for r_x and r_y. 
Note: For using the randgen function, uncomment 1,2,3 and comment 4,5 as given below
      For using the numpy random object, comment 1,2,3 and uncomment 4,5 as given below
'''
def get_ids(n1,n2,seed=0):
    global rndness
    numpy.random.seed(seed)
#     y=[
#     [
#     [0.18,0.18],[0.82,0.82]    # represent a contrast of 70%
#     ],
#     [
#     [0.18,0.50],[0.50,0.18],[0.5,0.5],[0.5,0.82],[0.82,0.5]   #represent a contrast of 50%
#     ],
#     [
#     [0.18,0.82],[0.82,0.18]   # represent a contrast of 30%
#     ]
#     ]
    z=numpy.zeros([n2,n1,2])
    
    if(rndness==[]):                  #1
        rndness = randgen(n1*n2)      #2
    for i in range(0,n2):
        for j in range(0,n1):
            type=rndness[i*n1 + j]    #3
            #type = numpy.random.choice([1,2,3,4,5,6,7,8,9],p=[(1/6),(1/15),(1/6),(1/15),(1/15),(1/15),(1/6),(1/15),(1/6)])   #4
            #rndness.append(type)       #5
            if(type==1):
                z[i][j] = [0.18,0.18]
            elif(type==2):
                z[i][j] = [0.18,0.50]
            elif(type==3):
                z[i][j] = [0.18,0.82]
            elif(type==4):
                z[i][j] = [0.5,0.18]
            elif(type==5):
                z[i][j] = [0.5,0.5]
            elif(type==6):
                z[i][j] = [0.5,0.82]
            elif(type==7):
                z[i][j] = [0.82,0.18]
            elif(type==8):
                z[i][j] = [0.82,0.5]
            else:
                z[i][j] = [0.82,0.82]
#             ii=random.randint(0,2)
#             z[i][j]=y[ii][random.randint(0,len(y[ii])-1)]
              
    return z

#turtle.clear()
#turtle.reset()
eps = 50
l=5760#float(input()) # length of the pattern
b=11520#float(input()) # breadth of the pattern
N=10#int(input())# number of sub-blocks in a block 
num_x=15#int(input())# number of super-blocks in x direction
num_y=30#int(input())# number of super-blocks in y direction
turtle.setworldcoordinates(0,b,l,0)
indexs=get_ids(num_x,num_y)
print(indexs)
x=0
y=0
col1=[['midnight blue','red']]
col2=[['white']]
board=turtle.Turtle(visible=False)
#board.reset()
board.hideturtle()
board.speed(0)
turtle.tracer(1000,1000)
turtle.setworldcoordinates(0,b,l,0)
forgloves = [['midnight blue','white'],['red','white']]
for i in range(0,num_y):
    for j in range(0,num_x):
#        block(board,x,y,(l/num_x)-eps,(b/num_y)-eps,N,indexs[i][j][0],indexs[i][j][1],forgloves[(i+j)%2])
        superblock(board,x,y,(l/num_x)-eps,(b/num_y)-eps,N,indexs[i][j][0],indexs[i][j][1],col1[(i+j)%len(col1)],col2[(i+j)%len(col2)])
#         if(i==num_y-1 and j==num_x-1):
#             print("made the last superblock")
        #draw_box(board,x+(l/num_x)-(2/3)*eps,0,eps/3,b,'black')
        x=x+(l/num_x)
    
    x=0
    #draw_box(board,0,y+(l/num_y)-(2/3)*eps,l,eps/3,'black')
    y=y+(b/num_y)
#x = l, y = b
superblock(board,x-(l/num_x),y-(b/num_y),(l/num_x),(b/num_y),N,indexs[num_y-1][num_x-1][0],indexs[num_y-1][num_x-1][1],col1[(num_y+num_x-2)%len(col1)],col2[(num_y+num_x-2)%len(col2)])
board.ht()
turtle.getcanvas().postscript(file='/home/cse/btech/cs1170355/scratch/'+str(l)+"_"+str(b)+"_"+str(N)+"_"+str(num_x)+"_"+str(num_y)+'wpad.ps')
f = open( '/home/cse/btech/cs1170355/scratch/parameters', 'w' )
f.write( 'parameters = ' + repr(indexs) + '\n' )
f.close()
f2 = open( '/home/cse/btech/cs1170355/scratch/sequence', 'w' )

#x=0;y=0
print(rndness)
for ji in range(len(rndness)):
    f2.write(str(rndness[ji]));f2.write(", ")
f2.close


# In[25]:


#col1[0][0]


# In[214]:


#Test randomness when randgen() is used

# print(len(rndness))
# counter=0
# for number in range(1,10):
#     counter=0
#     for ier in rndness:
#         if(ier==number):
#             counter+=1
#     if number in [1,3,7,9]:
#         print(number," ",counter/len(rndness)," ",0.16)
#     else:
#         print(number," ",counter/len(rndness)," ",0.06)


# In[219]:


#Test randomness when numpy random function is used

# print(len(rndness))
# counter2=0
# for number in range(1,10):
#     counter2=0
#     for ier in rndness:
#         if(ier==number):
#             counter2+=1
#     if number in [1,3,7,9]:
#         print(number," ",counter2/len(rndness)," ",0.16)
#     else:
#         print(number," ",counter2/len(rndness)," ",0.06)


