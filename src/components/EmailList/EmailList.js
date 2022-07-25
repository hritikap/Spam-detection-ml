import React from 'react';
import styles from './EmailList.module.css';
import { IconButton } from '@material-ui/core';
import InboxIcon from '@material-ui/icons/Inbox';
import RefreshIcon from '@material-ui/icons/Refresh';
import Section from '../Section/Section';
import ReactTooltip from 'react-tooltip';
import EmailRow from '../EmailRow/EmailRow';
import Loading from '../Loading/Loading';

function EmailList({
  emails,
  selectedLabelItem,
  setSelectedLabelItem,
  getMails,
}) {
  return (
    <div className={styles.emailList}>
      <div className={styles.emailList__settings}>
        <div className={styles.emailList__settingsLeft}>
          <ReactTooltip place='bottom' />
          <span data-tip='Refresh'>
            <IconButton onClick={getMails}>
              <RefreshIcon />
            </IconButton>
          </span>
        </div>
      </div>

      <div className={styles.emailList__sections}>
        {[{ icon: InboxIcon, title: 'Primary', color: 'red' }].map(
          (obj, index) => {
            return (
              <Section
                Icon={obj.icon}
                title={obj.title}
                color={obj.color}
                selected={selectedLabelItem === index}
                onClick={() => setSelectedLabelItem(index)}
              />
            );
          },
        )}
      </div>

      <div className={styles.emailList__list}>
        {emails.length === 0 ? (
          <Loading />
        ) : (
          <div>
            {emails.map(({ id, data }) => {
              return (
                <EmailRow
                  id={id}
                  key={id}
                  title={data.from}
                  subject={data.subject}
                  to={data.to}
                  from={data.from}
                  description={data.message}
                  time={new Date(data.timestamp?.seconds * 1000).toUTCString()}
                  read={data.read || false}
                  spam={data.spam || false}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default EmailList;
