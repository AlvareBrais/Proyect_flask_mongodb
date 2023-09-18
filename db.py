import random

class Jugador:
    def __init__(self, nombre):
        self.nombre = nombre
        # Fijar una semilla aleatoria basada en el nombre del jugador
        self.semilla = hash(nombre)  # Utilizamos hash() para obtener un número único basado en el nombre
        random.seed(self.semilla)
        
        self.elo = random.choice(["Hierro", "Bronze", "Plata", "Oro", "Platino", "Diamante", "Maestro", "Gran Maestro", "Challenger"])
        self.posicion = random.choice(["Top", "Jungla", "Mid", "Adc", "Support"])
        self.npartidas = random.randint(50, 350)
        self.winrate = round(random.uniform(10, 85), 2)
        self.puntos = random.randint(0, 100)

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
    jugador = Jugador(nombre)
    mycol.update_one({"nombre": nombre}, {"$set": jugador.__dict__}, upsert=True)

print("Datos insertados en MongoDB sin duplicados por el campo 'nombre'.")
