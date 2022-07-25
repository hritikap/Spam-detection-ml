import React, { useEffect, useState } from 'react';
import './App.css';
import Header from './components/Header/Header';
import Sidebar from './components/Sidebar/Sidebar';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import Mail from './components/Mail/Mail';
import EmailList from './components/EmailList/EmailList';
import SendMail from './components/SendMail/SendMail';
import Login from './components/Login/Login';
import { selectSendMessageIsOpen } from './features/mail';
import { useSelector, useDispatch } from 'react-redux';
import { selectUser, login } from './features/userSlice';
import { auth, db } from './firebase';

import { selectShowSidebar } from './features/commonSlice';
import { getQueryStatement, processMailData } from './utilities/utils';

function App() {
  const sendMessageIsOpen = useSelector(selectSendMessageIsOpen);
  const user = useSelector(selectUser);
  const showSideBar = useSelector(selectShowSidebar);
  const dispatch = useDispatch();

  const [emails, setEmails] = useState([]);
  const [selectedSideBarItem, setSelectedSideBarItem] = useState(0); //0->inbox,1->sent,2->spam
  const [selectedLabelItem, setSelectedLabelItem] = useState(0); // 0->primary
  const [emailReff, setEmailReff] = useState(null);
  const [searchQuery, setSearchQuery] = useState();

  const getMails = () => {
    let emailRef = getQueryStatement(selectedSideBarItem, selectedLabelItem);
    setEmailReff(emailRef);
  };

  //  useeffeect  is a react hook  which tell React that your component needs to do something after render.
  useEffect(() => {
    if (searchQuery && searchQuery.length > 0) {
      const m = db
        .collection('emails')
        .where('to', '==', auth.currentUser.email)
        .orderBy('timestamp', 'desc')
        .onSnapshot((snapshot) => {
          setEmails(snapshot.docs.map((doc) => processMailData(doc)));
        });

      // snapshot is the copy of the data a certain location that means uusng the abve code the data oor message that is composed is copied to the inbox of the receiver.

      return () => {
        console.log('search clean up');
        m();
      };
    } else if (emailReff) {
      console.log('----emailReff=> about to snapshot');
      const m = emailReff.onSnapshot((snapshot) => {
        setEmails([...snapshot.docs.map((doc) => processMailData(doc))]);
      });

      return () => {
        console.log('----emailReff=> calling cleanup');
        m();
        setEmails([]);
      };
    }
  }, [emailReff, searchQuery]);

  // to handle the login page using google authentication
  useEffect(() => {
    auth.onAuthStateChanged((user) => {
      if (user) {
        dispatch(
          login({
            displayName: user.displayName,
            email: user.email,
            photoUrl: user.photoURL,
          }),
        );
        getMails();
      }
    });
  }, [selectedSideBarItem, selectedLabelItem]);

  return (
    <Router>
      {!user ? (
        <Login />
      ) : (
        <div className='app'>
          <Header searchQuery={searchQuery} setSearchQuery={setSearchQuery} />

          <div className='app__body'>
            {showSideBar && (
              <Sidebar
                listLength={emails.length}
                selectedSideBarItem={selectedSideBarItem}
                setSelectedSideBarItem={setSelectedSideBarItem}
              />
            )}

            {/* The switch component looks through all of its child routes and it displays the first one whose path matches the current URL */}

            <Switch>
              <Route path='/mail'>
                <Mail />
              </Route>

              <Route path='/'>
                <EmailList
                  emails={emails}
                  selectedLabelItem={selectedLabelItem}
                  setSelectedLabelItem={setSelectedLabelItem}
                  getMails={getMails}
                />
              </Route>
            </Switch>
          </div>

          {sendMessageIsOpen && <SendMail />}
        </div>
      )}
    </Router>
  );
}

export default App;
