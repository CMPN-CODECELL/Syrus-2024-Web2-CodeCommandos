import React from "react";
import styled from "styled-components";
import {
  FaPlaneArrival as Trip,
  FaHome as Feed,
  FaSearch as Search,
  FaPaperPlane as DM,
  FaCog as Settings,
} from "react-icons/fa";
export default function ProfileButtons({changeMenu,setFeed}) {
  return (
    <Container>
      <div className="home button">
        <Feed className="icon" size="27"/>
        <h2>Home</h2>
      </div>
      <div className="search button" onClick={()=>setFeed("search")}>
        <Search className="icon" size="27"/>
        <h2>Search</h2>
      </div>
      <div className="mytrips button" onClick={()=>setFeed("trip")}>
        <Trip className="icon" size="27"/>
        <h2>My Trips</h2>
      </div>
      <div className="DM button" onClick={()=>changeMenu("Direct")}>
        <DM className="icon" size="27"/>
        <h2>Direct</h2>
      </div>
      <div className="Settings button" onClick={()=>changeMenu("Settings")}>
        <Settings className="icon" size="27"/>
        <h2>Settings</h2>
      </div>
    </Container>
  );
}
const Container = styled.div`
  padding-top: 2rem;
  gap: 2rem;
  width: 100%;
  display: flex;
  flex-direction: column;
  border-bottom: 0.05rem solid white;
  div {
    gap: 2rem;
    display: flex;
    flex-direction: row;
    h2,.icon{
      color: white;
    }
  }
  .button{
    cursor: pointer;
  }
`;
