const sqlite3 = require('sqlite3').verbose();  
const db = new sqlite3.Database('./seresInmortales.db');  

// Crear la tabla si no existe y agregar datos iniciales  
db.serialize(() => {  
    db.run(`CREATE TABLE IF NOT EXISTS seres (  
        id INTEGER PRIMARY KEY AUTOINCREMENT,  
        name TEXT,  
        age INTEGER,  
        enroll BOOLEAN  
    )`, (err) => {  
        if (err) {  
            console.error("Error creando la tabla:", err);  
        }  
    });  
});  

module.exports = db;
