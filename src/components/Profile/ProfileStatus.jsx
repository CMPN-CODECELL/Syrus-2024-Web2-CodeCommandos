import React from "react";
import styled from "styled-components";
export default function ProfileStatus({ changeMenu, setSearchWhat}) {
  return (
    <Container>
      <div className="row posts" onClick={()=>changeMenu("Posts")}>
        <h2>46</h2>
        <h4>Posts</h4>
      </div>
      <div
        className="row followers"
        onClick={() => {
          setSearchWhat("Followers")
          changeMenu("Search");
        }}
      >
        <h2>46</h2>
        <h4>Followers</h4>
      </div>
      <div
        className="row following"
        onClick={() => {
          setSearchWhat("Following")
          changeMenu("Search");
        }}
      >
        <h2>46</h2>
        <h4>Following</h4>
      </div>
    </Container>
  );
}
const Container = styled.div`
  width: 100%;
  display: grid;
  grid-template-columns: 33% 33% 34%;
  align-items: center;
  user-select: none;
  h2 {
    color: white;
  }
  h4 {
    color: #978f8f;
  }
  .row {
    border-left: 0.1rem solid;
    display: flex;
    flex-direction: column;
    align-items: center;
    border-right: 0.1rem solid;
  }
  .posts,
  .followers,
  .following {
    cursor: pointer;
    border-right: none;
  }
  .posts {
    border-left: none;
  }
`;
