from utils import *
def chat():
    nam = "Unknown"
    # call = multiFunctionCall({"whoIs":whoIs,"emo":emo, "identifyu":identifyu, "whathappen":whathappen, "learnq":learnq, "learna":learna})
    firstQuestion="Hi, I am chatbot."
    template = "Unknown.template"
    #print(template)
    decryp(template)
    
    Chat(template).converse(firstQuestion)
    
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

chat()
