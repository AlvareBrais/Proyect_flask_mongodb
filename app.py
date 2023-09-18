from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import json

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Cambia esto por una clave secreta segura

# Configuración de la base de datos MongoDB
client = MongoClient('mongodb://localhost:27017')  # Conexión local a MongoDB
mydb = client['Mongodb']  # Nombre de la base de datos

# Crear una colección llamada "users" si no existe
if 'users' not in mydb.list_collection_names():
    mydb.create_collection('users')

# Obtener la colección "users"
users_collection = mydb['users']

mycol = mydb['Jugadores']

# Ruta para el registro de usuarios
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Capturar los datos del formulario
        username = request.form['username']
        password = request.form['password']

        # Verificar si el usuario ya existe
        existing_user = users_collection.find_one({'username': username})
        if existing_user:
            return "El usuario ya existe. Por favor, elige otro nombre de usuario."

        # Hash de la contraseña antes de guardarla
        hashed_password = generate_password_hash(password, method='sha256')

        # Crear un nuevo documento de usuario
        new_user = {
            'username': username,
            'password': hashed_password
        }

        # Insertar el nuevo usuario en la colección "users"
        users_collection.insert_one(new_user)

        return "Registro exitoso. Ahora puedes iniciar sesión."

    return render_template('signup.html')

# Ruta para el inicio de sesión
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Capturar los datos del formulario
        username = request.form['username']
        password = request.form['password']

        # Buscar al usuario en la base de datos
        user = users_collection.find_one({'username': username})

        if user and check_password_hash(user['password'], password):
            # Iniciar sesión del usuario
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return "Nombre de usuario o contraseña incorrectos."

    return render_template('login.html')

# Ruta principal (requiere inicio de sesión)
@app.route('/')
def index():
    if 'username' in session:
        # Obtener los jugadores o datos que deseas mostrar
        jugadores = mycol.find({})  # Cambia esto según tus necesidades
        return render_template('index.html', jugadores=jugadores)
    else:
        return redirect(url_for('login'))

# Ruta para cerrar sesión
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
