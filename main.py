from utils import *
def chat():
    nam = identifyu()
    call = multiFunctionCall({"whoIs":whoIs,"emo":emo, "identifyu":identifyu, "whathappen":whathappen, "learnq":learnq, "learna":learna})
    if nam == "Unknown":
        firstQuestion="Hi, I am chatbot."
        template = "Unknown.template"
    else :
        firstQuestion="Hi "+nam+" , nice to see you again."
        template = nam+".template"
    #print(template)
    decryp(template)
    
    Chat(template, reflections,call=call).converse(firstQuestion)
    
    from os import path
    if path.exists(nam+".txt"):
        with open(nam+".txt", "r") as myfile:
            daa = myfile.read()
            with open(nam+".template", "a") as myf:
                now = str(datetime.now())
                myf.write("\n{ mood "+now+": "+sas(daa)+" }")
                myf.write("\n{ reason "+now+": "+daa+" }")
        os.remove(nam+".txt")
    
    if path.exists("learn.txt"):
        with open("learn.txt", "r") as myfile:
            daa = myfile.read()
            with open(nam+".template", "a") as myf:
                myf.write(daa)
        os.remove("learn.txt")
        
    encryp(template)


# In[73]:


chat()