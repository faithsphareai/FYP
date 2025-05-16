import { useSQLiteContext } from 'expo-sqlite';

export function useDatabase() {
  const db = useSQLiteContext();
  
  // Initialize database tables
  const initializeTables = async () => {
    try {
      await db.execAsync(`
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          email TEXT NOT NULL,
          password TEXT NOT NULL
        )
      `);
      return true;
    } catch (error) {
      console.error("Database initialization error:", error);
      return false;
    }
  };
  
  // Create a new user
  const createUser = async (name, email, password) => {
    try {
      await db.runAsync(
        'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
        [name, email, password]
      );
      return { success: true };
    } catch (error) {
      console.error("Error creating user:", error);
      return { success: false, error };
    }
  };
  
  // Find user by email
  const findUserByEmail = async (email) => {
    try {
      const result = await db.getAllAsync(
        'SELECT * FROM users WHERE email = ?',
        [email]
      );
      return result.length > 0 ? result[0] : null;
    } catch (error) {
      console.error("Error finding user:", error);
      return null;
    }
  };
  
  // Authenticate user
  const authenticateUser = async (email, password) => {
    try {
      const user = await findUserByEmail(email);
      
      if (!user) {
        return { success: false, message: 'User not found' };
      }
      
      if (user.password !== password) {
        return { success: false, message: 'Incorrect password' };
      }
      
      return { success: true, user };
    } catch (error) {
      console.error("Authentication error:", error);
      return { success: false, message: 'Authentication failed' };
    }
  };
  
  return {
    initializeTables,
    createUser,
    findUserByEmail,
    authenticateUser
  };
}