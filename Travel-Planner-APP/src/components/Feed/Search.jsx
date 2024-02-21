import React from "react";
import styled from "styled-components";
import Hotel from "./Hotel";
import Flight from "./Flight";
import Activity from "./Activity";
import Suggestion from "./Suggestion";
export default function Search() {
  return <Container>
    <Hotel/>
    <Flight/>
    <Suggestion/>
  </Container>;
}
const Container = styled.div`
  overflow: hidden;
  height: 100%;
  width: 100%;
  display: grid;
  grid-template-columns: 33% 33% 33%;
`;
