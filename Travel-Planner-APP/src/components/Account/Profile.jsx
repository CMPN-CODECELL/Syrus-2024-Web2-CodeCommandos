import React from "react";
import styled from "styled-components";
import ProfileLogo from "../Profile/ProfileLogo";
import ProfileInfo from "../Profile/ProfileInfo";
import ProfileButtons from "../Profile/ProfileButtons";
import Logout from "../Profile/Logout";
export default function Profile({changeMenu,setFeed}) {
  return (<Container>
    <ProfileLogo/>
    <ProfileInfo changeMenu={changeMenu}/>
    <ProfileButtons changeMenu={changeMenu} setFeed={setFeed}/>
    <Logout/>
  </Container>);
}
const Container = styled.div`
  height: 100vh;
  width: 100%;
  display: grid;
  grid-template-rows: 5% 30% 50% 15%;
  padding: 1.5rem 1.5rem 1.5rem 1.5rem;
  background-color: #282a34;
`;
