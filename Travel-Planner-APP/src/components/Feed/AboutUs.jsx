import React from "react";
import gaurav from "../../assets/homelander.jpg";
import toshit from "../../assets/toshith.jpg";
import surya from "../../assets/surya.jpg";
import shreyas from "../../assets/shreyas.jpg";
import styled from "styled-components";
export default function AboutUs() {
  return <Container>
    <body>
    <div class="rectangle">
    <h2>About Us</h2>
    <div class="vision">
        <h1>Our Vision</h1>
        <section id="vision">
            <p>At Travel Companion, we envision a world where travel is not just a journey from one place to another, but a transformative experience that leaves a lasting impact on every traveler. Our vision is to redefine the way people explore, connect, and create memories by providing innovative and personalized travel solutions.</p>
    
            <p>We aspire to break barriers, inspire wanderlust, and empower individuals to embark on journeys that are as unique as they are. Through cutting-edge technology and a deep understanding of user needs, we aim to simplify and enhance the entire travel experience.</p>
    
            <p>As we look forward, our vision encompasses creating a global community of passionate travelers who share unforgettable adventures, foster meaningful connections, and contribute to a more interconnected and understanding world.</p>
    
            <p>Join us on this exciting journey as we strive to make every travel dream a reality, one personalized itinerary at a time.</p>
        </section>
        </div> 
    </div>
    <div class="gap">
        <div class="static-txt">We</div>
        <ul class="dynamic-txts">
            <li><span>Explore</span></li>
            <li><span>Discover</span></li>
            <li><span>Experience</span></li>
            <li><span>Enjoy</span></li>
            <li><span>Share</span></li>
        </ul>
    </div>
    <h1 class="header">Meet our Team</h1>
    <div class="team">
    <div className="team">
    <div className="box box1">
        <div className="img img1"></div>
        <h3 className="name">Shreyas Bagwe </h3>
        <div className="icons">
            <a target="_main" href="https://www.linkedin.com/in/shreyas-bagwe-12a666269/"><i className="fa-brands fa-linkedin"></i></a>
            <a target="_main" href="https://www.instagram.com/shreyas_1015/"><i className="fa-brands fa-instagram"></i></a>
            <a target="_main" href="https://github.com/Shreyasb1015"><i className="fa-brands fa-github"></i></a>
        </div>
    </div>
    <div className="box box2">
        <div className="img img2"></div>
        <h3 className="name">Gaurav Mahadeshwar</h3>
        <div className="icons">
            <a target="_main" href="https://www.linkedin.com/in/gaurav-mahadeshwar-a6178a257/"><i className="fa-brands fa-linkedin"></i></a>
            <a target="_main" href="https://www.instagram.com/gaurav_rm11/"><i className="fa-brands fa-instagram"></i></a>
            <a target="_main" href="https://github.com/gaurav-rm11"><i className="fa-brands fa-github"></i></a>
        </div>
    </div>
    <div className="box box3">
        <div className="img img3"></div>
        <h3 className="name">Toshit Hole</h3>
        <div className="icons">
            <a target="_main" href="https://www.linkedin.com/in/toshit-d-h-a2607b23b/"><i className="fa-brands fa-linkedin"></i></a>
            <a target="_main" href="https://www.instagram.com/toshit_dh/"><i className="fa-brands fa-instagram"></i></a>
            <a target="_main" href="https://github.com/toshit-dh"><i className="fa-brands fa-github"></i></a>
        </div>
    </div>
    <div className="box box4">
        <div className="img img4"></div>
        <h3 className="name">Surayanarayan</h3>
        <div className="icons">
            <a target="_main" href="https://www.linkedin.com/in/suryanarayan-panigrahi-9b8807226/"><i className="fa-brands fa-linkedin"></i></a>
            <a target="_main" href="https://www.instagram.com/suryanarayan_5294/"><i className="fa-brands fa-instagram"></i></a>
            <a target="_main" href="https://github.com/Surya17847"><i className="fa-brands fa-github"></i></a>
        </div>
    </div>
</div></div>
</body>
  </Container>;
}
const Container = styled.div`
  height: 100%;
  width: 100%;
  body {
    background-color: #070f2b;
    font-family: sans-serif;
    padding: 0%;
    margin: 0%;
  }
  .rectangle {
    width: 100%;
    height: 550px;
    background: rgb(238, 174, 202);
    background: radial-gradient(
      circle,
      rgba(238, 174, 202, 1) 0%,
      rgba(148, 187, 233, 1) 100%
    ); /* background-image: url('https://images.unsplash.com/photo-1504732099162-d8c9d5ba3bfd?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'); */
    background-position: center;
    background-size: cover;
    clip-path: polygon(
      100% 0,
      100% 35%,
      100% 100%,
      83% 82%,
      66% 100%,
      49% 82%,
      32% 100%,
      17% 80%,
      0 100%,
      0 0
    );
  }
  h2 {
    color: #070f2b;
    position: absolute;
    top: 10px;
    left: 40vw;
    font-size: 50px;
  }

  h2:hover {
    box-shadow: 0px 0px 5px 5px inset;
    cursor: pointer;
  }
  .vision {
    padding-left: 20px;
    position: absolute;
    top: 90px;
  }
  .gap {
    height: 550px;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .team {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 80px;
    border-bottom: 10px;
  }
  .box {
    height: 450px;
    width: 290px;
    border: 4px solid;
    border-color: transparent;
    border-radius: 5%;
    margin-left: 30px;
    margin-right: 20px;
    transition: box-shadow 0.3s ease;
  }
  .box > .name {
    color: #3498db;
    text-align: center;
    top: 50%;
    font-size: 25px;
  }
  .icons {
    margin-left: 60px;
  }
  .icons > i {
    margin-left: 20px;
    margin-top: -10px;
    font-size: 30px;
    color: #1b1a55;
    text-decoration: none;
  }
  .box > .icons > a {
    text-decoration: none;
    color: #1b1a55;
    display: inline-block;
    cursor: pointer;
    margin-left: 20px;
    margin-top: -10px;
    font-size: 30px;
  }
  a:visited {
    color: #1b1a55;
    margin-left: 20px;
    margin-top: -10px;
    font-size: 30px;
  }

  .img {
    height: 70%;
    width: 100%;
  }
  .header {
    position: relative;
    top: 40px;
    font-size: 45px;
    color: #9290c3;
    left: 35vw;
  }
  .gap {
    display: flex;
    flex-direction: column;
    max-width: 600px;
    justify-content: center;
    align-items: center;
    text-align: center;
    margin: 20px auto;
    padding-left: 60px;
    font-size: 90px;
    height: 600px;
  }

  .gap .static-txt {
    padding-top: 250px;
    margin-left: -680px;
    color: #9290c3;
    font-size: 90px;
    font-weight: 400;
    margin-bottom: 10px;
  }

  .gap .dynamic-txts {
    margin-top: -110px;
    height: 90px;
    line-height: 90px;
    overflow: hidden;
  }
  .dynamic-txts li {
    list-style: none;
    color: #111a6d;
    font-size: 100px;
    font-weight: 450;
    position: relative;
    top: 0;
    animation: slide 12s steps(4) infinite;
  }

  @keyframes slide {
    100% {
      top: -360px;
    }
  }

  .dynamic-txts li span {
    position: relative;
    margin: 5px 0;
    line-height: 90px;
  }
  .dynamic-txts li span::after {
    content: "";
    position: absolute;
    left: 0;
    height: 100%;
    width: 100%;
    border-left: 2px solid #111a6d;
    animation: typing 3s steps(10) infinite;
  }
  @keyframes typing {
    40%,
    60% {
      left: calc(100% + 30px);
    }
    100% {
      left: 0;
    }
  }
  @keyframes glow {
    0% {
      box-shadow: 0 0 10px #3498db; /* Initial box-shadow */
    }
    50% {
      box-shadow: 0 0 20px #3498db; /* Intermediate box-shadow */
    }
    100% {
      box-shadow: 0 0 10px #3498db; /* Final box-shadow */
    }
  }
  .box:hover {
    animation: glow 1s infinite;
  }
  .img1 {
    background-image: url("shreyas.jpg");
    background-size: cover;
    border-top-left-radius: 5%;
    border-top-right-radius: 5%;
  }
  .img2 {
    background-image: url("homelander.jpg");
    background-size: cover;
    border-top-left-radius: 5%;
    border-top-right-radius: 5%;
  }
  .img3 {
    background-image: url("toshith.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    border-top-left-radius: 5%;
    border-top-right-radius: 5%;
  }
  .img4 {
    background-image: url("surya.jpg");
    background-size: cover;
    border-top-left-radius: 5%;
    border-top-right-radius: 5%;
  }

  /* For screens smaller than 600px */
  @media (max-width: 600px) {
    .rectangle {
      h2 {
        font-size: 40px;
        top: -20px;
        left: 100px;
      }
      .vision {
        top: 60px;
      }
      height: 800px;
      margin-bottom: 30px;
      padding-bottom: 20px;
    }

    .vision {
      top: 5%;
    }

    .gap {
      font-size: 8vw;
      height: auto;
    }

    .gap .static-txt {
      margin-left: -10%;
      font-size: 8vw;
    }

    .dynamic-txts li {
      font-size: 8vw;
    }

    .dynamic-txts li span {
      margin: 2% 0;
      line-height: 8vw;
    }

    .dynamic-txts li span::after {
      border-left: 2vw solid #111a6d;
    }

    .box {
      width: 80vw;
      height: 400px;
      margin: 5% auto;
    }

    .box > .name {
      font-size: 30px;
    }

    .icons {
      margin-left: 5%;
      display: flex;
      justify-content: center;
      align-items: center;
    }

    .icons > i {
      margin-left: 2%;
      margin-top: -2%;
      font-size: 30px;
      left: 5px;
      right: 10px;
    }

    .img {
      height: 70%;
    }

    .header {
      font-size: 7vw;
      left: 80px;
    }
  }

  /* For screens between 601px and 1024px */
  @media (min-width: 601px) and (max-width: 1024px) {
    .rectangle {
      height: auto;
    }

    .vision {
      top: 5%;
    }

    .gap {
      font-size: 5vw;
      height: auto;
    }

    .gap .static-txt {
      margin-left: -8%;
      font-size: 5vw;
    }

    .dynamic-txts li {
      font-size: 5vw;
    }

    .dynamic-txts li span {
      margin: 1% 0;
      line-height: 5vw;
    }

    .dynamic-txts li span::after {
      border-left: 1vw solid #111a6d;
    }

    .box {
      width: 70vw;
      height: auto;
      margin: 5% auto;
    }

    .box > .name {
      font-size: 3vw;
    }

    .icons {
      margin-left: 8%;
    }

    .icons > i {
      margin-left: 2%;
      margin-top: -1%;
      font-size: 3vw;
    }

    .img {
      height: 40vw;
    }

    .header {
      font-size: 4vw;
    }
  }

  /* Additional styles for screens smaller than 1024px */
  @media (max-width: 1024px) {
    .team {
      flex-direction: column;
      align-items: center;
    }

    .box {
      width: 80vw;
      margin: 5% auto;
    }
  }
`;
