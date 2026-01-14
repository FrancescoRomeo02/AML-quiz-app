// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBjr_QQIREjqVrcNrqeylElwUNVaZ0sY58",
  authDomain: "aml-quiz-ff6bf.firebaseapp.com",
  projectId: "aml-quiz-ff6bf",
  storageBucket: "aml-quiz-ff6bf.firebasestorage.app",
  messagingSenderId: "40765590972",
  appId: "1:40765590972:web:b9c441c1bd18031842efb9",
  measurementId: "G-TLFSLMC3VW"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize and Export Services
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
export const db = getFirestore(app);