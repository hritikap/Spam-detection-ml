import React from 'react';
import styles from './Sidebar.module.css';
import AddIcon from '@material-ui/icons/Add';
import InboxIcon from '@material-ui/icons/Inbox';
import { Button } from '@material-ui/core';
import SidebarOption from '../SidebarOption/SidebarOption';
import NearMeIcon from '@material-ui/icons/NearMe';
import NewReleasesIcon from '@material-ui/icons/NewReleases';
import { openSendMessage } from '../../features/mail';
import { useDispatch } from 'react-redux';

function Sidebar({ selectedSideBarItem, setSelectedSideBarItem, listLength }) {
  const dispatch = useDispatch();

  return (
    <div className={styles.sidebar}>
      <Button
        name='composebtnK'
        startIcon={<AddIcon fontsize='large' />}
        className={styles.sidebar_compose}
        onClick={() => dispatch(openSendMessage())}
      >
        Compose
      </Button>

      <SidebarOption
        setSelectedSideBarItem={setSelectedSideBarItem}
        index={0}
        selected={selectedSideBarItem === 0}
        Icon={InboxIcon}
        title='Inbox'
        number={selectedSideBarItem === 0 ? listLength : ''}
      />

      <SidebarOption
        setSelectedSideBarItem={setSelectedSideBarItem}
        index={4}
        selected={selectedSideBarItem === 4}
        Icon={NearMeIcon}
        title='Sent'
        number={selectedSideBarItem === 4 ? listLength : ''}
      />

      <SidebarOption
        setSelectedSideBarItem={setSelectedSideBarItem}
        index={6}
        selected={selectedSideBarItem === 6}
        Icon={NewReleasesIcon}
        title='Spam'
        number={selectedSideBarItem === 6 ? listLength : ''}
      />
    </div>
  );
}

export default Sidebar;
