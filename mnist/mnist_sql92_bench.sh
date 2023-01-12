#!/bin/bash

# parameters
parallel=${PARALLEL:-"20"}
lr=0.2 # learningrate
attss="20 200"
iters="1 10 100"
limits="2 20 200 2000"
repeat=3

echo "
create table mnist (label float, pixel1 float, pixel2 float, pixel3 float, pixel4 float, pixel5 float, pixel6 float, pixel7 float, pixel8 float, pixel9 float, pixel10 float, pixel11 float, pixel12 float, pixel13 float, pixel14 float, pixel15 float, pixel16 float, pixel17 float, pixel18 float, pixel19 float, pixel20 float, pixel21 float, pixel22 float, pixel23 float, pixel24 float, pixel25 float, pixel26 float, pixel27 float, pixel28 float, pixel29 float, pixel30 float, pixel31 float, pixel32 float, pixel33 float, pixel34 float, pixel35 float, pixel36 float, pixel37 float, pixel38 float, pixel39 float, pixel40 float, pixel41 float, pixel42 float, pixel43 float, pixel44 float, pixel45 float, pixel46 float, pixel47 float, pixel48 float, pixel49 float, pixel50 float, pixel51 float, pixel52 float, pixel53 float, pixel54 float, pixel55 float, pixel56 float, pixel57 float, pixel58 float, pixel59 float, pixel60 float, pixel61 float, pixel62 float, pixel63 float, pixel64 float, pixel65 float, pixel66 float, pixel67 float, pixel68 float, pixel69 float, pixel70 float, pixel71 float, pixel72 float, pixel73 float, pixel74 float, pixel75 float, pixel76 float, pixel77 float, pixel78 float, pixel79 float, pixel80 float, pixel81 float, pixel82 float, pixel83 float, pixel84 float, pixel85 float, pixel86 float, pixel87 float, pixel88 float, pixel89 float, pixel90 float, pixel91 float, pixel92 float, pixel93 float, pixel94 float, pixel95 float, pixel96 float, pixel97 float, pixel98 float, pixel99 float, pixel100 float, pixel101 float, pixel102 float, pixel103 float, pixel104 float, pixel105 float, pixel106 float, pixel107 float, pixel108 float, pixel109 float, pixel110 float, pixel111 float, pixel112 float, pixel113 float, pixel114 float, pixel115 float, pixel116 float, pixel117 float, pixel118 float, pixel119 float, pixel120 float, pixel121 float, pixel122 float, pixel123 float, pixel124 float, pixel125 float, pixel126 float, pixel127 float, pixel128 float, pixel129 float, pixel130 float, pixel131 float, pixel132 float, pixel133 float, pixel134 float, pixel135 float, pixel136 float, pixel137 float, pixel138 float, pixel139 float, pixel140 float, pixel141 float, pixel142 float, pixel143 float, pixel144 float, pixel145 float, pixel146 float, pixel147 float, pixel148 float, pixel149 float, pixel150 float, pixel151 float, pixel152 float, pixel153 float, pixel154 float, pixel155 float, pixel156 float, pixel157 float, pixel158 float, pixel159 float, pixel160 float, pixel161 float, pixel162 float, pixel163 float, pixel164 float, pixel165 float, pixel166 float, pixel167 float, pixel168 float, pixel169 float, pixel170 float, pixel171 float, pixel172 float, pixel173 float, pixel174 float, pixel175 float, pixel176 float, pixel177 float, pixel178 float, pixel179 float, pixel180 float, pixel181 float, pixel182 float, pixel183 float, pixel184 float, pixel185 float, pixel186 float, pixel187 float, pixel188 float, pixel189 float, pixel190 float, pixel191 float, pixel192 float, pixel193 float, pixel194 float, pixel195 float, pixel196 float, pixel197 float, pixel198 float, pixel199 float, pixel200 float, pixel201 float, pixel202 float, pixel203 float, pixel204 float, pixel205 float, pixel206 float, pixel207 float, pixel208 float, pixel209 float, pixel210 float, pixel211 float, pixel212 float, pixel213 float, pixel214 float, pixel215 float, pixel216 float, pixel217 float, pixel218 float, pixel219 float, pixel220 float, pixel221 float, pixel222 float, pixel223 float, pixel224 float, pixel225 float, pixel226 float, pixel227 float, pixel228 float, pixel229 float, pixel230 float, pixel231 float, pixel232 float, pixel233 float, pixel234 float, pixel235 float, pixel236 float, pixel237 float, pixel238 float, pixel239 float, pixel240 float, pixel241 float, pixel242 float, pixel243 float, pixel244 float, pixel245 float, pixel246 float, pixel247 float, pixel248 float, pixel249 float, pixel250 float, pixel251 float, pixel252 float, pixel253 float, pixel254 float, pixel255 float, pixel256 float, pixel257 float, pixel258 float, pixel259 float, pixel260 float, pixel261 float, pixel262 float, pixel263 float, pixel264 float, pixel265 float, pixel266 float, pixel267 float, pixel268 float, pixel269 float, pixel270 float, pixel271 float, pixel272 float, pixel273 float, pixel274 float, pixel275 float, pixel276 float, pixel277 float, pixel278 float, pixel279 float, pixel280 float, pixel281 float, pixel282 float, pixel283 float, pixel284 float, pixel285 float, pixel286 float, pixel287 float, pixel288 float, pixel289 float, pixel290 float, pixel291 float, pixel292 float, pixel293 float, pixel294 float, pixel295 float, pixel296 float, pixel297 float, pixel298 float, pixel299 float, pixel300 float, pixel301 float, pixel302 float, pixel303 float, pixel304 float, pixel305 float, pixel306 float, pixel307 float, pixel308 float, pixel309 float, pixel310 float, pixel311 float, pixel312 float, pixel313 float, pixel314 float, pixel315 float, pixel316 float, pixel317 float, pixel318 float, pixel319 float, pixel320 float, pixel321 float, pixel322 float, pixel323 float, pixel324 float, pixel325 float, pixel326 float, pixel327 float, pixel328 float, pixel329 float, pixel330 float, pixel331 float, pixel332 float, pixel333 float, pixel334 float, pixel335 float, pixel336 float, pixel337 float, pixel338 float, pixel339 float, pixel340 float, pixel341 float, pixel342 float, pixel343 float, pixel344 float, pixel345 float, pixel346 float, pixel347 float, pixel348 float, pixel349 float, pixel350 float, pixel351 float, pixel352 float, pixel353 float, pixel354 float, pixel355 float, pixel356 float, pixel357 float, pixel358 float, pixel359 float, pixel360 float, pixel361 float, pixel362 float, pixel363 float, pixel364 float, pixel365 float, pixel366 float, pixel367 float, pixel368 float, pixel369 float, pixel370 float, pixel371 float, pixel372 float, pixel373 float, pixel374 float, pixel375 float, pixel376 float, pixel377 float, pixel378 float, pixel379 float, pixel380 float, pixel381 float, pixel382 float, pixel383 float, pixel384 float, pixel385 float, pixel386 float, pixel387 float, pixel388 float, pixel389 float, pixel390 float, pixel391 float, pixel392 float, pixel393 float, pixel394 float, pixel395 float, pixel396 float, pixel397 float, pixel398 float, pixel399 float, pixel400 float, pixel401 float, pixel402 float, pixel403 float, pixel404 float, pixel405 float, pixel406 float, pixel407 float, pixel408 float, pixel409 float, pixel410 float, pixel411 float, pixel412 float, pixel413 float, pixel414 float, pixel415 float, pixel416 float, pixel417 float, pixel418 float, pixel419 float, pixel420 float, pixel421 float, pixel422 float, pixel423 float, pixel424 float, pixel425 float, pixel426 float, pixel427 float, pixel428 float, pixel429 float, pixel430 float, pixel431 float, pixel432 float, pixel433 float, pixel434 float, pixel435 float, pixel436 float, pixel437 float, pixel438 float, pixel439 float, pixel440 float, pixel441 float, pixel442 float, pixel443 float, pixel444 float, pixel445 float, pixel446 float, pixel447 float, pixel448 float, pixel449 float, pixel450 float, pixel451 float, pixel452 float, pixel453 float, pixel454 float, pixel455 float, pixel456 float, pixel457 float, pixel458 float, pixel459 float, pixel460 float, pixel461 float, pixel462 float, pixel463 float, pixel464 float, pixel465 float, pixel466 float, pixel467 float, pixel468 float, pixel469 float, pixel470 float, pixel471 float, pixel472 float, pixel473 float, pixel474 float, pixel475 float, pixel476 float, pixel477 float, pixel478 float, pixel479 float, pixel480 float, pixel481 float, pixel482 float, pixel483 float, pixel484 float, pixel485 float, pixel486 float, pixel487 float, pixel488 float, pixel489 float, pixel490 float, pixel491 float, pixel492 float, pixel493 float, pixel494 float, pixel495 float, pixel496 float, pixel497 float, pixel498 float, pixel499 float, pixel500 float, pixel501 float, pixel502 float, pixel503 float, pixel504 float, pixel505 float, pixel506 float, pixel507 float, pixel508 float, pixel509 float, pixel510 float, pixel511 float, pixel512 float, pixel513 float, pixel514 float, pixel515 float, pixel516 float, pixel517 float, pixel518 float, pixel519 float, pixel520 float, pixel521 float, pixel522 float, pixel523 float, pixel524 float, pixel525 float, pixel526 float, pixel527 float, pixel528 float, pixel529 float, pixel530 float, pixel531 float, pixel532 float, pixel533 float, pixel534 float, pixel535 float, pixel536 float, pixel537 float, pixel538 float, pixel539 float, pixel540 float, pixel541 float, pixel542 float, pixel543 float, pixel544 float, pixel545 float, pixel546 float, pixel547 float, pixel548 float, pixel549 float, pixel550 float, pixel551 float, pixel552 float, pixel553 float, pixel554 float, pixel555 float, pixel556 float, pixel557 float, pixel558 float, pixel559 float, pixel560 float, pixel561 float, pixel562 float, pixel563 float, pixel564 float, pixel565 float, pixel566 float, pixel567 float, pixel568 float, pixel569 float, pixel570 float, pixel571 float, pixel572 float, pixel573 float, pixel574 float, pixel575 float, pixel576 float, pixel577 float, pixel578 float, pixel579 float, pixel580 float, pixel581 float, pixel582 float, pixel583 float, pixel584 float, pixel585 float, pixel586 float, pixel587 float, pixel588 float, pixel589 float, pixel590 float, pixel591 float, pixel592 float, pixel593 float, pixel594 float, pixel595 float, pixel596 float, pixel597 float, pixel598 float, pixel599 float, pixel600 float, pixel601 float, pixel602 float, pixel603 float, pixel604 float, pixel605 float, pixel606 float, pixel607 float, pixel608 float, pixel609 float, pixel610 float, pixel611 float, pixel612 float, pixel613 float, pixel614 float, pixel615 float, pixel616 float, pixel617 float, pixel618 float, pixel619 float, pixel620 float, pixel621 float, pixel622 float, pixel623 float, pixel624 float, pixel625 float, pixel626 float, pixel627 float, pixel628 float, pixel629 float, pixel630 float, pixel631 float, pixel632 float, pixel633 float, pixel634 float, pixel635 float, pixel636 float, pixel637 float, pixel638 float, pixel639 float, pixel640 float, pixel641 float, pixel642 float, pixel643 float, pixel644 float, pixel645 float, pixel646 float, pixel647 float, pixel648 float, pixel649 float, pixel650 float, pixel651 float, pixel652 float, pixel653 float, pixel654 float, pixel655 float, pixel656 float, pixel657 float, pixel658 float, pixel659 float, pixel660 float, pixel661 float, pixel662 float, pixel663 float, pixel664 float, pixel665 float, pixel666 float, pixel667 float, pixel668 float, pixel669 float, pixel670 float, pixel671 float, pixel672 float, pixel673 float, pixel674 float, pixel675 float, pixel676 float, pixel677 float, pixel678 float, pixel679 float, pixel680 float, pixel681 float, pixel682 float, pixel683 float, pixel684 float, pixel685 float, pixel686 float, pixel687 float, pixel688 float, pixel689 float, pixel690 float, pixel691 float, pixel692 float, pixel693 float, pixel694 float, pixel695 float, pixel696 float, pixel697 float, pixel698 float, pixel699 float, pixel700 float, pixel701 float, pixel702 float, pixel703 float, pixel704 float, pixel705 float, pixel706 float, pixel707 float, pixel708 float, pixel709 float, pixel710 float, pixel711 float, pixel712 float, pixel713 float, pixel714 float, pixel715 float, pixel716 float, pixel717 float, pixel718 float, pixel719 float, pixel720 float, pixel721 float, pixel722 float, pixel723 float, pixel724 float, pixel725 float, pixel726 float, pixel727 float, pixel728 float, pixel729 float, pixel730 float, pixel731 float, pixel732 float, pixel733 float, pixel734 float, pixel735 float, pixel736 float, pixel737 float, pixel738 float, pixel739 float, pixel740 float, pixel741 float, pixel742 float, pixel743 float, pixel744 float, pixel745 float, pixel746 float, pixel747 float, pixel748 float, pixel749 float, pixel750 float, pixel751 float, pixel752 float, pixel753 float, pixel754 float, pixel755 float, pixel756 float, pixel757 float, pixel758 float, pixel759 float, pixel760 float, pixel761 float, pixel762 float, pixel763 float, pixel764 float, pixel765 float, pixel766 float, pixel767 float, pixel768 float, pixel769 float, pixel770 float, pixel771 float, pixel772 float, pixel773 float, pixel774 float, pixel775 float, pixel776 float, pixel777 float, pixel778 float, pixel779 float, pixel780 float, pixel781 float, pixel782 float, pixel783 float, pixel784 float);
create table mnist2 (id int, label float, pixel1 float, pixel2 float, pixel3 float, pixel4 float, pixel5 float, pixel6 float, pixel7 float, pixel8 float, pixel9 float, pixel10 float, pixel11 float, pixel12 float, pixel13 float, pixel14 float, pixel15 float, pixel16 float, pixel17 float, pixel18 float, pixel19 float, pixel20 float, pixel21 float, pixel22 float, pixel23 float, pixel24 float, pixel25 float, pixel26 float, pixel27 float, pixel28 float, pixel29 float, pixel30 float, pixel31 float, pixel32 float, pixel33 float, pixel34 float, pixel35 float, pixel36 float, pixel37 float, pixel38 float, pixel39 float, pixel40 float, pixel41 float, pixel42 float, pixel43 float, pixel44 float, pixel45 float, pixel46 float, pixel47 float, pixel48 float, pixel49 float, pixel50 float, pixel51 float, pixel52 float, pixel53 float, pixel54 float, pixel55 float, pixel56 float, pixel57 float, pixel58 float, pixel59 float, pixel60 float, pixel61 float, pixel62 float, pixel63 float, pixel64 float, pixel65 float, pixel66 float, pixel67 float, pixel68 float, pixel69 float, pixel70 float, pixel71 float, pixel72 float, pixel73 float, pixel74 float, pixel75 float, pixel76 float, pixel77 float, pixel78 float, pixel79 float, pixel80 float, pixel81 float, pixel82 float, pixel83 float, pixel84 float, pixel85 float, pixel86 float, pixel87 float, pixel88 float, pixel89 float, pixel90 float, pixel91 float, pixel92 float, pixel93 float, pixel94 float, pixel95 float, pixel96 float, pixel97 float, pixel98 float, pixel99 float, pixel100 float, pixel101 float, pixel102 float, pixel103 float, pixel104 float, pixel105 float, pixel106 float, pixel107 float, pixel108 float, pixel109 float, pixel110 float, pixel111 float, pixel112 float, pixel113 float, pixel114 float, pixel115 float, pixel116 float, pixel117 float, pixel118 float, pixel119 float, pixel120 float, pixel121 float, pixel122 float, pixel123 float, pixel124 float, pixel125 float, pixel126 float, pixel127 float, pixel128 float, pixel129 float, pixel130 float, pixel131 float, pixel132 float, pixel133 float, pixel134 float, pixel135 float, pixel136 float, pixel137 float, pixel138 float, pixel139 float, pixel140 float, pixel141 float, pixel142 float, pixel143 float, pixel144 float, pixel145 float, pixel146 float, pixel147 float, pixel148 float, pixel149 float, pixel150 float, pixel151 float, pixel152 float, pixel153 float, pixel154 float, pixel155 float, pixel156 float, pixel157 float, pixel158 float, pixel159 float, pixel160 float, pixel161 float, pixel162 float, pixel163 float, pixel164 float, pixel165 float, pixel166 float, pixel167 float, pixel168 float, pixel169 float, pixel170 float, pixel171 float, pixel172 float, pixel173 float, pixel174 float, pixel175 float, pixel176 float, pixel177 float, pixel178 float, pixel179 float, pixel180 float, pixel181 float, pixel182 float, pixel183 float, pixel184 float, pixel185 float, pixel186 float, pixel187 float, pixel188 float, pixel189 float, pixel190 float, pixel191 float, pixel192 float, pixel193 float, pixel194 float, pixel195 float, pixel196 float, pixel197 float, pixel198 float, pixel199 float, pixel200 float, pixel201 float, pixel202 float, pixel203 float, pixel204 float, pixel205 float, pixel206 float, pixel207 float, pixel208 float, pixel209 float, pixel210 float, pixel211 float, pixel212 float, pixel213 float, pixel214 float, pixel215 float, pixel216 float, pixel217 float, pixel218 float, pixel219 float, pixel220 float, pixel221 float, pixel222 float, pixel223 float, pixel224 float, pixel225 float, pixel226 float, pixel227 float, pixel228 float, pixel229 float, pixel230 float, pixel231 float, pixel232 float, pixel233 float, pixel234 float, pixel235 float, pixel236 float, pixel237 float, pixel238 float, pixel239 float, pixel240 float, pixel241 float, pixel242 float, pixel243 float, pixel244 float, pixel245 float, pixel246 float, pixel247 float, pixel248 float, pixel249 float, pixel250 float, pixel251 float, pixel252 float, pixel253 float, pixel254 float, pixel255 float, pixel256 float, pixel257 float, pixel258 float, pixel259 float, pixel260 float, pixel261 float, pixel262 float, pixel263 float, pixel264 float, pixel265 float, pixel266 float, pixel267 float, pixel268 float, pixel269 float, pixel270 float, pixel271 float, pixel272 float, pixel273 float, pixel274 float, pixel275 float, pixel276 float, pixel277 float, pixel278 float, pixel279 float, pixel280 float, pixel281 float, pixel282 float, pixel283 float, pixel284 float, pixel285 float, pixel286 float, pixel287 float, pixel288 float, pixel289 float, pixel290 float, pixel291 float, pixel292 float, pixel293 float, pixel294 float, pixel295 float, pixel296 float, pixel297 float, pixel298 float, pixel299 float, pixel300 float, pixel301 float, pixel302 float, pixel303 float, pixel304 float, pixel305 float, pixel306 float, pixel307 float, pixel308 float, pixel309 float, pixel310 float, pixel311 float, pixel312 float, pixel313 float, pixel314 float, pixel315 float, pixel316 float, pixel317 float, pixel318 float, pixel319 float, pixel320 float, pixel321 float, pixel322 float, pixel323 float, pixel324 float, pixel325 float, pixel326 float, pixel327 float, pixel328 float, pixel329 float, pixel330 float, pixel331 float, pixel332 float, pixel333 float, pixel334 float, pixel335 float, pixel336 float, pixel337 float, pixel338 float, pixel339 float, pixel340 float, pixel341 float, pixel342 float, pixel343 float, pixel344 float, pixel345 float, pixel346 float, pixel347 float, pixel348 float, pixel349 float, pixel350 float, pixel351 float, pixel352 float, pixel353 float, pixel354 float, pixel355 float, pixel356 float, pixel357 float, pixel358 float, pixel359 float, pixel360 float, pixel361 float, pixel362 float, pixel363 float, pixel364 float, pixel365 float, pixel366 float, pixel367 float, pixel368 float, pixel369 float, pixel370 float, pixel371 float, pixel372 float, pixel373 float, pixel374 float, pixel375 float, pixel376 float, pixel377 float, pixel378 float, pixel379 float, pixel380 float, pixel381 float, pixel382 float, pixel383 float, pixel384 float, pixel385 float, pixel386 float, pixel387 float, pixel388 float, pixel389 float, pixel390 float, pixel391 float, pixel392 float, pixel393 float, pixel394 float, pixel395 float, pixel396 float, pixel397 float, pixel398 float, pixel399 float, pixel400 float, pixel401 float, pixel402 float, pixel403 float, pixel404 float, pixel405 float, pixel406 float, pixel407 float, pixel408 float, pixel409 float, pixel410 float, pixel411 float, pixel412 float, pixel413 float, pixel414 float, pixel415 float, pixel416 float, pixel417 float, pixel418 float, pixel419 float, pixel420 float, pixel421 float, pixel422 float, pixel423 float, pixel424 float, pixel425 float, pixel426 float, pixel427 float, pixel428 float, pixel429 float, pixel430 float, pixel431 float, pixel432 float, pixel433 float, pixel434 float, pixel435 float, pixel436 float, pixel437 float, pixel438 float, pixel439 float, pixel440 float, pixel441 float, pixel442 float, pixel443 float, pixel444 float, pixel445 float, pixel446 float, pixel447 float, pixel448 float, pixel449 float, pixel450 float, pixel451 float, pixel452 float, pixel453 float, pixel454 float, pixel455 float, pixel456 float, pixel457 float, pixel458 float, pixel459 float, pixel460 float, pixel461 float, pixel462 float, pixel463 float, pixel464 float, pixel465 float, pixel466 float, pixel467 float, pixel468 float, pixel469 float, pixel470 float, pixel471 float, pixel472 float, pixel473 float, pixel474 float, pixel475 float, pixel476 float, pixel477 float, pixel478 float, pixel479 float, pixel480 float, pixel481 float, pixel482 float, pixel483 float, pixel484 float, pixel485 float, pixel486 float, pixel487 float, pixel488 float, pixel489 float, pixel490 float, pixel491 float, pixel492 float, pixel493 float, pixel494 float, pixel495 float, pixel496 float, pixel497 float, pixel498 float, pixel499 float, pixel500 float, pixel501 float, pixel502 float, pixel503 float, pixel504 float, pixel505 float, pixel506 float, pixel507 float, pixel508 float, pixel509 float, pixel510 float, pixel511 float, pixel512 float, pixel513 float, pixel514 float, pixel515 float, pixel516 float, pixel517 float, pixel518 float, pixel519 float, pixel520 float, pixel521 float, pixel522 float, pixel523 float, pixel524 float, pixel525 float, pixel526 float, pixel527 float, pixel528 float, pixel529 float, pixel530 float, pixel531 float, pixel532 float, pixel533 float, pixel534 float, pixel535 float, pixel536 float, pixel537 float, pixel538 float, pixel539 float, pixel540 float, pixel541 float, pixel542 float, pixel543 float, pixel544 float, pixel545 float, pixel546 float, pixel547 float, pixel548 float, pixel549 float, pixel550 float, pixel551 float, pixel552 float, pixel553 float, pixel554 float, pixel555 float, pixel556 float, pixel557 float, pixel558 float, pixel559 float, pixel560 float, pixel561 float, pixel562 float, pixel563 float, pixel564 float, pixel565 float, pixel566 float, pixel567 float, pixel568 float, pixel569 float, pixel570 float, pixel571 float, pixel572 float, pixel573 float, pixel574 float, pixel575 float, pixel576 float, pixel577 float, pixel578 float, pixel579 float, pixel580 float, pixel581 float, pixel582 float, pixel583 float, pixel584 float, pixel585 float, pixel586 float, pixel587 float, pixel588 float, pixel589 float, pixel590 float, pixel591 float, pixel592 float, pixel593 float, pixel594 float, pixel595 float, pixel596 float, pixel597 float, pixel598 float, pixel599 float, pixel600 float, pixel601 float, pixel602 float, pixel603 float, pixel604 float, pixel605 float, pixel606 float, pixel607 float, pixel608 float, pixel609 float, pixel610 float, pixel611 float, pixel612 float, pixel613 float, pixel614 float, pixel615 float, pixel616 float, pixel617 float, pixel618 float, pixel619 float, pixel620 float, pixel621 float, pixel622 float, pixel623 float, pixel624 float, pixel625 float, pixel626 float, pixel627 float, pixel628 float, pixel629 float, pixel630 float, pixel631 float, pixel632 float, pixel633 float, pixel634 float, pixel635 float, pixel636 float, pixel637 float, pixel638 float, pixel639 float, pixel640 float, pixel641 float, pixel642 float, pixel643 float, pixel644 float, pixel645 float, pixel646 float, pixel647 float, pixel648 float, pixel649 float, pixel650 float, pixel651 float, pixel652 float, pixel653 float, pixel654 float, pixel655 float, pixel656 float, pixel657 float, pixel658 float, pixel659 float, pixel660 float, pixel661 float, pixel662 float, pixel663 float, pixel664 float, pixel665 float, pixel666 float, pixel667 float, pixel668 float, pixel669 float, pixel670 float, pixel671 float, pixel672 float, pixel673 float, pixel674 float, pixel675 float, pixel676 float, pixel677 float, pixel678 float, pixel679 float, pixel680 float, pixel681 float, pixel682 float, pixel683 float, pixel684 float, pixel685 float, pixel686 float, pixel687 float, pixel688 float, pixel689 float, pixel690 float, pixel691 float, pixel692 float, pixel693 float, pixel694 float, pixel695 float, pixel696 float, pixel697 float, pixel698 float, pixel699 float, pixel700 float, pixel701 float, pixel702 float, pixel703 float, pixel704 float, pixel705 float, pixel706 float, pixel707 float, pixel708 float, pixel709 float, pixel710 float, pixel711 float, pixel712 float, pixel713 float, pixel714 float, pixel715 float, pixel716 float, pixel717 float, pixel718 float, pixel719 float, pixel720 float, pixel721 float, pixel722 float, pixel723 float, pixel724 float, pixel725 float, pixel726 float, pixel727 float, pixel728 float, pixel729 float, pixel730 float, pixel731 float, pixel732 float, pixel733 float, pixel734 float, pixel735 float, pixel736 float, pixel737 float, pixel738 float, pixel739 float, pixel740 float, pixel741 float, pixel742 float, pixel743 float, pixel744 float, pixel745 float, pixel746 float, pixel747 float, pixel748 float, pixel749 float, pixel750 float, pixel751 float, pixel752 float, pixel753 float, pixel754 float, pixel755 float, pixel756 float, pixel757 float, pixel758 float, pixel759 float, pixel760 float, pixel761 float, pixel762 float, pixel763 float, pixel764 float, pixel765 float, pixel766 float, pixel767 float, pixel768 float, pixel769 float, pixel770 float, pixel771 float, pixel772 float, pixel773 float, pixel774 float, pixel775 float, pixel776 float, pixel777 float, pixel778 float, pixel779 float, pixel780 float, pixel781 float, pixel782 float, pixel783 float, pixel784 float);
create table if not exists img (i int, j int, v float);
create table if not exists one_hot(i int, j int, v int, dummy int);
copy mnist from './mnist_train.csv' delimiter ',';
insert into mnist2 (select row_number() over (), * from mnist limit 1000);
"
for i in `seq 1 784`; do
echo "insert into img (select id,$i,pixel$i/255 from mnist2);"
done
echo "
insert into one_hot(select n.i, n.j, coalesce(i.v,0), i.v from (select id,label+1 as label,1 as v from mnist2) i right outer join (select a.a as i, b.b as j from (select generate_series as a from generate_series(1,60000)) a, (select generate_series as b from generate_series(1,10)) b) n on n.i=i.id and n.j=i.label order by i,j);

create table if not exists w_xh (w_id int, i int, j int, v float);
create table if not exists w_ho (w_id int, i int, j int, v float);

"
for atts in $attss; do
echo "
insert into w_xh (select $atts, i.*,j.*,random()*2-1 from generate_series(1,784) i, generate_series(1,$atts) j);
insert into w_ho (select $atts, i.*,j.*,random()*2-1 from generate_series(1,$atts) i, generate_series(1,10) j);
"
done
for limit in $limits; do
#  for iter in $iters; do
iter=$((6000/limit))
    for atts in $attss; do
echo "
\record gd_mnist.csv Umbra-SQL-92,$atts,$limit,$lr,$iter,$parallel
with recursive w (iter,id,i,j,v) as (
  (select 0,0,i,j,v from w_xh where w_id=$atts union select 0,1,i,j,v from w_ho where w_id=$atts)
  union all
  (
  with w_now as (
     SELECT * from w
  ), a_xh(i,j,v) as (
     SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v)))
     FROM (select * from img) AS m INNER JOIN w_now AS n ON m.j=n.i
     WHERE n.id=0 and n.iter=(select max(iter) from w_now) -- w_xh
     GROUP BY m.i, n.j
  ), a_ho(i,j,v) as (
     SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v)))
     FROM a_xh AS m INNER JOIN w_now AS n ON m.j=n.i
     WHERE n.id=1 and n.iter=(select max(iter) from w_now)  -- w_ho
     GROUP BY m.i, n.j
  ), l_ho(i,j,v) as (
     select m.i, m.j, 2*(m.v-n.v)
     from a_ho AS m INNER JOIN one_hot AS n ON m.i=n.i AND m.j=n.j
  ), d_ho(i,j,v) as (
     select m.i, m.j, m.v*n.v*(1-n.v)
     from l_ho AS m INNER JOIN a_ho AS n ON m.i=n.i AND m.j=n.j
  ), l_xh(i,j,v) as (
     SELECT m.i, n.i as j, (SUM (m.v*n.v)) -- transpose
     FROM d_ho AS m INNER JOIN w_now AS n ON m.j=n.j
     WHERE n.id=1 and n.iter=(select max(iter) from w_now)  -- w_ho
     GROUP BY m.i, n.i
  ), d_xh(i,j,v) as (
     select m.i, m.j, m.v*n.v*(1-n.v)
     from l_xh AS m INNER JOIN a_xh AS n ON m.i=n.i AND m.j=n.j
  ), d_w(id,i,j,v) as (
     SELECT 0, m.j as i, n.j, (SUM (m.v*n.v))
     FROM (select * from img) AS m INNER JOIN d_xh AS n ON m.i=n.i
     GROUP BY m.j, n.j
     union
     SELECT 1, m.j as i, n.j, (SUM (m.v*n.v))
     FROM a_xh AS m INNER JOIN d_ho AS n ON m.i=n.i
     GROUP BY m.j, n.j
  )
  select iter+1, w.id, w.i, w.j, w.v - $lr * d_w.v
  from w_now as w, d_w
  where iter < $iter and w.id=d_w.id and w.i=d_w.i and w.j=d_w.j
  )
)
select iter, count(*) from w group by iter order by iter;
"
done
done
#done
