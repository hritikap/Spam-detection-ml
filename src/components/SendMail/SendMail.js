import React, { useState } from 'react';
import styles from './SendMail.module.css';
import CloseIcon from '@material-ui/icons/Close';

import { Button } from '@material-ui/core';
import { useForm } from 'react-hook-form';
import { useDispatch } from 'react-redux';
import { closeSendMessage } from '../../features/mail';
import { auth, db } from '../../firebase';
import firebase from 'firebase';
import { CKEditor } from '@ckeditor/ckeditor5-react';
import ClassicEditor from '@ckeditor/ckeditor5-build-classic';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { encrypt } from '../../utilities/crypt';
import { generateRoomName } from '../../utilities/common';
import axios from 'axios';

function SendMail() {
  const { register, handleSubmit, errors } = useForm();
  const dispatch = useDispatch();
  const [addData, setVal] = useState('');
  const [option] = useState('Primary');

  const sendEmail = async () => {
    console.log('addData');
    let cleanMsg = addData.replace(/(<([^>]+)>)/gi, '');

    let config = {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    };

    const resp = await axios.post(
      'http://127.0.0.1:5000/predict',
      { message: cleanMsg },
      config,
    );
    return resp.data['val'];
  };

  const handleChange = (e, editor) => {
    var data = editor.getData();
    setVal(data);
  };

  // const handleChangeinType = (event) => {
  //   setOption(event.target.value);
  //   console.log(`Option selected:`, option);
  // };

  const checkIfEmailExists = async (email) => {
    const snapshot = await db
      .collection('users')
      .where('email', '==', email)
      .limit(1)
      .get();
    console.log(snapshot.empty);

    if (snapshot.empty || email === auth.currentUser.email) {
      return false;
    }
    return true;
  };

  const onSubmit = async (formData) => {
    // check here if email exist (for now just setting it to true)
    if (addData === '') {
      return;
    }
    const emailExists = await checkIfEmailExists(formData.to);

    // we can Compose and send an email based on the contents of a document written to a specified Cloud Firestore collection.

    //when we authenticate a user using fireabase google authentication the firebase provides a built in email system and a goofle smtp server.

    //when we send an email in the firestore database it stores the sender,recever,meesages,spamlabel and the time of thee message sent
    if (emailExists) {
      db.collection('emails').add({
        to: formData.to,
        from: auth.currentUser.email,
        subject: encrypt(
          formData.subject,
          generateRoomName(auth.currentUser.email, formData.to),
        ),
        message: encrypt(
          addData,
          generateRoomName(auth.currentUser.email, formData.to),
        ),
        timestamp: firebase.firestore.FieldValue.serverTimestamp(),

        read: false,

        spam: await sendEmail(formData.message),
        label: option,
      });
      //  useDispatch hook is used to dispatch an action
      dispatch(closeSendMessage());
      alert('Mail sent successfully to  ' + formData.to);
    } else {
      console.log(formData.to + " doesn't exist.");
      toast.error('Cannot send mail to ' + formData.to);
    }
  };

  return (
    <>
      {/* toast container to notify whether the mail has been sent or not */}
      <ToastContainer />
      <div className={styles.sendMail}>
        <div className={styles.sendMail__header}>
          <h3>New Mail</h3>
          <CloseIcon
            className={styles.sendMail__close}
            onClick={() => dispatch(closeSendMessage())}
          />
        </div>

        {/* handlesubmit is the part of react hook form  */}

        <form onSubmit={handleSubmit(onSubmit)}>
          <input
            name='to'
            placeholder='To'
            // type='text'
            type='email'
            ref={register({ required: true })}
          />
          {errors.to && (
            <p className={styles.sendMail__error}>To is required</p>
          )}
          <input
            name='subject'
            placeholder='Subject'
            type='text'
            ref={register({ required: true })}
          />
          {errors.to && (
            <p className={styles.sendMail__error}>Subject is required</p>
          )}

          <CKEditor
            editor={ClassicEditor}
            // styles={{overflow:"auto",back}}
            id='body_ckeditor'
            data={addData}
            onChange={handleChange}
          />

          {errors.to && (
            <p className={styles.sendMail__error}>Message is required</p>
          )}

          <div className={styles.sendMail__buttons}>
            <Button
              className='sendMail__send'
              variant='contained'
              color='primary'
              type='submit'
              name='mailClick'
            >
              Send
            </Button>
          </div>
        </form>
      </div>
    </>
  );
}

export default SendMail;
