import pymongo
import random
import json

random.seed(42)
# Conectarse a MongoDB
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["Mongodb"]
mycol = mydb["Jugadores"]

# Lista de nombres
nombres = ["GankYouVeryMuch", "AFKorFeed", "TowerDiveExpert", "CtrlAltDefeat", "LeonaDelRey", "BaronForBreakfast",
           "FeedMeMoreELO", "WardrobeMalfunction", "MyNexusYourNexus", "HookedOnThresh", "MissFortunate",
           "LeagueOfLegendsAnonymous", "NerfMyLife", "NotoriousGANK", "YOLOqueue", "AsheToMalph", "GG_EZ_Game",
           "DontBlameTeemo", "JunglingForDummies", "SixItemsNoWards", "TheFlashGordon", "JhinAndTonic", "CarryMePlz",
           "PentaStealer", "LAGtastic", "DiscoNunu", "InELOHell", "UrgotToBeKidding", "YasuoMainBTW", "RitoPlease",
           "RoamingForKills", "SmiteThee", "MinionSlayer", "BlueBuffBandit", "ILoveFarm", "LeashMeUp",
           "VisionIsForTryhards", "NerfThisChamp", "Ganktopus", "IMainTeemo", "KatarinaTheGreat", "JustCantEven",
           "RunningOutOfMana", "CaughtByMum", "PingSpikes", "MemeLordAndSavior"]

# Insertar documentos en MongoDB evitando duplicados por el campo "nombre"
for nombre in nombres:
    jugador = {
        "nombre": nombre,
        "elo": random.choice(["Hierro", "Bronze", "Plata", "Oro", "Platino", "Diamante", "Maestro", "Gran Maestro", "Challenger"]),
        "posicion": random.choice(["Top", "Jungla", "Mid", "Adc", "Support"]),
        "npartidas": random.randint(50, 350),
        "winrate": round(random.uniform(10, 85), 2),
        "puntos": random.randint(0, 100)
    }
    # Intentar insertar el documento, evitando duplicados por el campo "nombre"
    mycol.update_one({"nombre": nombre}, {"$set": jugador}, upsert=True)

print("Datos insertados en MongoDB sin duplicados por el campo 'nombre'.")
