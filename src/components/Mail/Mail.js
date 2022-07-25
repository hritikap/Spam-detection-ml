import React, { useEffect, useState, useRef } from 'react';
import styles from './Mail.module.css';
import { IconButton } from '@material-ui/core';
import ArrowBackIcon from '@material-ui/icons/ArrowBack';
import NewReleasesIcon from '@material-ui/icons/NewReleases';
import { useHistory } from 'react-router-dom';
import { selectOpenMail } from '../../features/mailSlice';
import { useSelector } from 'react-redux';
import { auth, db } from '../../firebase';
import ReactHtmlParser from 'react-html-parser';
import { toggleSpam } from '../../utilities/utils';
import Loading from '../Loading/Loading';
import ReactTooltip from 'react-tooltip';

import DoneAllOutlinedIcon from '@material-ui/icons/DoneAllOutlined';

function Mail() {
  const history = useHistory(); //The useHistory hook allows us to access React Router's history object.Through the history object, we can access and manipulate the current state of the browser history.

  const selectedMail = useSelector(selectOpenMail); //useSelector hook is used to get the state from the redux store.useSelector is a function that takes the current state as an argument and returns whatever data you want from it.

  const [spam, setSpam] = useState(false);
  const componentRef = useRef();

  useEffect(() => {
    if (!selectedMail) {
      history.push('/');
      return;
    }

    //  If receiver open this mail, set its read status as true
    if (selectedMail.from !== auth.currentUser.email) {
      db.collection('emails').doc(selectedMail.id).set(
        {
          read: true,
        },
        { merge: true },
      );
    }

    setSpam(selectedMail.spam || false);
  }, []);

  return !selectedMail ? (
    <Loading />
  ) : (
    <div className={styles.mail}>
      <div className={styles.mail__tools}>
        <div className={styles.mail__toolsLeft}>
          <ReactTooltip place='bottom' />
          <p data-tip='Back'>
            <IconButton onClick={() => history.push('/')}>
              <ArrowBackIcon />
            </IconButton>
          </p>

          <ReactTooltip place='bottom' />
          <p data-tip='Report As Spam'>
            <IconButton
              onClick={async (e) => {
                e.stopPropagation();
                const result = await toggleSpam(selectedMail.id);
                if (result) setSpam(!spam);
              }}
            >
              {spam ? (
                <NewReleasesIcon style={{ fill: 'red' }} />
              ) : (
                <NewReleasesIcon />
              )}
            </IconButton>
          </p>
        </div>
      </div>

      <div
        className={styles.mail__body}
        style={{ position: 'relative' }}
        ref={componentRef}
      >
        <div style={{ position: 'absolute', top: '0px', right: '20px' }}>
          {selectedMail && selectedMail.read
            ? selectedMail.from === auth.currentUser.email && (
                <div style={{ padding: '10px' }}>
                  <ReactTooltip place='left' />
                  <span data-tip='Read by recipient'>
                    <DoneAllOutlinedIcon style={{ color: 'blue' }} />
                  </span>
                </div>
              )
            : selectedMail.from === auth.currentUser.email && (
                <div style={{ padding: '10px' }}>
                  {/* Unread by recipient */}
                  <ReactTooltip place='left' />
                  <span data-tip='Delivered to recipient'>
                    <DoneAllOutlinedIcon style={{ color: 'darkgray' }} />
                  </span>
                </div>
              )}
        </div>

        <div className={styles.mail__bodyHeader}>
          <h2>{selectedMail?.subject}</h2>

          <p>{selectedMail?.title}</p>
          <p className={styles.mail__time}>{selectedMail?.time}</p>
        </div>
        <div className={styles.mail__message}>
          <p>{ReactHtmlParser(selectedMail?.description)}</p>
          <br></br>
        </div>
        <br></br>
      </div>
    </div>
  );
}

export default Mail;
