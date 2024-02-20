import React, { useEffect, useState } from "react";
import styled from "styled-components";
import Profile from "./Account/Profile";
import { useNavigate } from "react-router-dom";
import ProfileDirect from "./Account/ProfileDirect";
import ProfileSettings from "./Account/ProfileSettings";
import Messages from "./Account/Messages";
import EditProfile from "./Account/EditProfile";
export default function Account({ messages,setMessages,changeMenu,menu,setFeed}) {
  const navigate = useNavigate()
  useEffect(()=>{
    const user = JSON.parse(localStorage.getItem('user'))
    if(!user) navigate('/auth')
  },[])
  const {user} = JSON.parse(localStorage.getItem('user'))
  switch (menu) {
    case "Direct":
      return (
        <Container>
          <ProfileDirect changeMenu={changeMenu} setMessages={setMessages}/>
        </Container>
      );
    case "Settings":
      return (
        <Container>
          <ProfileSettings changeMenu={changeMenu} />
        </Container>
      );
    case "Profile":
      return (
        <Container>
          <Profile changeMenu={changeMenu} setFeed={setFeed}/>
        </Container>
      );
    case "Messages":
      return (
        <Container>
          <Messages changeMenu={changeMenu} messages={messages}/>
        </Container>
      );
    case "Edit Profile":
      return (
        <Container>
          <EditProfile changeMenu={changeMenu} user={user}/>
        </Container>
      );
      
  }
}
const Container = styled.div`
  height: 100vh;
  width: 100%;
  background-color: #282a34;
`;
