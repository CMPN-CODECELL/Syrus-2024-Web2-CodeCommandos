import React from "react";
import styled from "styled-components";
import Posts from "../Feed/Posts";

export default function UserPosts({changeMenu}) {
  return <Container>
    <Posts inUser={true} changeMenu={changeMenu}/>
  </Container>;
}
const Container = styled.div`
  height: 140%;
  width: 100%;
`;
