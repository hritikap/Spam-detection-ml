import firebase from 'firebase';

const firebaseConfig = {
  apiKey: 'AIzaSyAS_7GhYzdRu3Cqp0BfV0213yge01kMsiM',
  authDomain: 'spamdetectormail-16f12.firebaseapp.com',
  projectId: 'spamdetectormail-16f12',
  storageBucket: 'spamdetectormail-16f12.appspot.com',
  messagingSenderId: '339961846529',
  appId: '1:339961846529:web:ee654f4af77027e1d9d900',
  measurementId: 'G-8FTZ24Y1VZ',
};

const firebaseApp = firebase.initializeApp(firebaseConfig);
const db = firebase.firestore();
const auth = firebase.auth();
const provider = new firebase.auth.GoogleAuthProvider();

export { db, auth, provider };
