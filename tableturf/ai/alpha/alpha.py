import logging
from typing import List, Optional

import cv2
import numpy as np

from logger import logger
from tableturf.ai import AI
from tableturf.ai.alpha import util
from tableturf.model import Card, Stage, Status, Step, Grid


class Alpha(AI):
    # @util.pause
    def redraw(self, hands: List[Card], stage: Optional[Stage] = None, my_deck: Optional[List[Card]] = None, his_deck: Optional[List[Card]] = None) -> bool:
        return self.__score_redraw(hands, stage, my_deck, his_deck) < 0

    @staticmethod
    def __score_redraw(hands: List[Card], stage: Optional[Stage] = None, my_deck: Optional[List[Card]] = None, his_deck: Optional[List[Card]] = None) -> float:
        hand_quantile = 2  # [0, 3]
        deck_quantile = 8  # [0, 14]
        cards = sorted([card for card in hands], key=lambda c: c.size)
        if my_deck is None:
            return 1 if cards[hand_quantile].size >= 10 else -1
        deck = sorted([card for card in hands + my_deck], key=lambda c: c.size)

        if logger.isEnabledFor(logging.DEBUG):
            hand_sizes = [card.size for card in cards]
            deck_sizes = [card.size for card in deck]
            logger.debug(f'tableturf.ai.alpha.redraw: hand_sizes={hand_sizes}, deck_sizes={deck_sizes}')

        if cards[hand_quantile].size >= deck[deck_quantile].size:
            if np.all([util.is_special_card(card) for card in cards[hand_quantile:]]):
                return -1
            else:
                return 1
        else:
            return -1

    # @util.pause
    def next_step(self, status: Status) -> Step:
        logger.debug(f'tableturf.ai.alpha.next_step: round={status.round}')
        if status.round > 2:
            remaining_cost = sorted([card.sp_cost for card in status.hands + status.my_deck], reverse=True)
            sp_threshold = min(np.sum(remaining_cost[:2]), 6)
            # only can drop cards or special attack
            steps = status.get_possible_steps(action=Step.Action.Place)
            if len(steps) == 0 or status.my_sp >= sp_threshold:
                steps = status.get_possible_steps()
                scores = np.array([self.__score_special_attack_step(status, step, sp_threshold) for step in steps])
                logger.debug(f'tableturf.ai.alpha.next_step.special_attack: scores={scores}, steps={steps}, case=special_attack')
                return steps[np.argmax(np.sum(scores, axis=1))]

            if status.round >= 5:
                # try to expand
                current_score = self.__score_current_stage(status)
                cards = self.__sort_hands_for_expanding(status)
                logger.debug(f'tableturf.ai.alpha.next_step.expanding: cards={cards}, case=expanding')
                for card in cards:
                    steps = status.get_possible_steps(card, action=Step.Action.Place)
                    scores = np.array([self.__score_expanding_step(current_score, status, step) for step in steps])
                    logger.debug(f'tableturf.ai.alpha.next_step.expanding: scores={scores}, steps={steps}, case=expanding')
                    if self.__is_good_for_expanding(scores, steps):
                        return steps[np.argmax(np.sum(scores, axis=1))]

            # try to consolidate
            remaining_sp_card = len([card.sp_cost for card in status.hands + status.my_deck if util.is_special_card(card)])
            steps = status.get_possible_steps(action=Step.Action.Place)
            scores = np.array([self.__score_consolidating_step(status, step, remaining_sp_card) for step in steps])
            logger.debug(f'tableturf.ai.alpha.next_step.consolidating: scores={scores}, steps={steps}, case=consolidating')
            return steps[np.argmax(np.sum(scores, axis=1))]

        elif status.round == 2:
            result = dict()
            for card in status.hands:
                steps = status.get_possible_steps(card)
                scores = np.array([self.__score_round_2_step(status, step) for step in steps])
                possible_sp = np.unique(scores[:, 2])
                for sp in possible_sp:
                    sub_group = scores[:, 2] == sp
                    sub_steps = np.array(steps)[sub_group]
                    sub_scores = scores[sub_group]
                    step = sub_steps[np.argmax(np.sum(sub_scores[:, :2], axis=1))]
                    next_status = util.estimate_status(status, step, expand=True)[0]
                    # pick the max one who can use special attack
                    _cards = [c for c in next_status.hands if len(next_status.get_possible_steps(c)) > 1]
                    if len(_cards) == 0:
                        continue
                    _card = max(_cards, key=lambda c: c.size)
                    _steps = next_status.get_possible_steps(_card)
                    _scores = np.array([self.__score_round_1_step(next_status, _step) for _step in _steps])
                    result[step] = np.max(np.sum(_scores, axis=1))
            logger.debug(f'tableturf.ai.alpha.next_step.round_2: result={result}, case=round_2')
            if len(result) == 0:
                return status.get_possible_steps()[0]
            return max(result, key=result.get)

        elif status.round == 1:
            steps = status.get_possible_steps()
            scores = np.array([self.__score_round_1_step(status, step) for step in steps])
            logger.debug(f'tableturf.ai.alpha.next_step.round_1: scores={scores}, steps={steps}, case=round_1')
            return steps[np.argmax(np.sum(scores, axis=1))]

        logger.error(f'tableturf.ai.alpha.next_step: unexpected behavior')
        return status.get_possible_steps()[0]  # should not be here

    @staticmethod
    def __score_current_stage(status: Status):
        occupied_grids_1 = Evaluation.occupied_grids(status.stage, my_dilate=1, his_dilate=1, connectivity=8)
        occupied_grids_2 = Evaluation.occupied_grids(status.stage, my_dilate=2, his_dilate=1, connectivity=8)
        occupied_grids_3 = Evaluation.occupied_grids(status.stage, my_dilate=3, his_dilate=1, connectivity=8)
        conflict_grids = Evaluation.conflict_grids(status.stage, my_dilate=3, his_dilate=3)
        return occupied_grids_1, occupied_grids_2, occupied_grids_3, conflict_grids

    @staticmethod
    def __score_expanding_step(current_score, status: Status, step: Step):
        occupied_grids_1, occupied_grids_2, occupied_grids_3, conflict_grids = current_score
        next_stage = util.estimate_stage(status.stage, step)
        estimated_occupied_grids_1 = Evaluation.occupied_grids(next_stage, my_dilate=1, his_dilate=1, connectivity=8) - occupied_grids_1
        estimated_occupied_grids_2 = Evaluation.occupied_grids(next_stage, my_dilate=2, his_dilate=1, connectivity=8) - occupied_grids_2
        estimated_occupied_grids_3 = Evaluation.occupied_grids(next_stage, my_dilate=3, his_dilate=1, connectivity=8) - occupied_grids_3
        estimated_conflict_grids = Evaluation.conflict_grids(next_stage, my_dilate=2, his_dilate=2) - conflict_grids
        size = Evaluation.ink_size(next_stage)
        my_sp = status.my_sp + util.estimate_my_sp_diff(status.stage, next_stage, step)
        his_sp_diff = util.estimate_his_sp_diff(status.stage, next_stage, step)

        pattern = step.card.get_pattern(step.rotate)
        distance = np.min([Evaluation.square_distance(status.stage, pos) for pos in pattern.offset + step.pos])
        if status.round >= 11 and distance > 2:
            distance = distance * -20
        else:
            distance = distance * -1

        return estimated_occupied_grids_1, estimated_occupied_grids_2 * 0.7, estimated_occupied_grids_3 * 0.5, estimated_conflict_grids * -1, distance, size, my_sp * 6, his_sp_diff * -4

    @staticmethod
    def __is_good_for_expanding(scores: np.ndarray, steps: List[Step]) -> bool:
        _idx = np.argmax(np.sum(scores[:, :3], axis=1))
        if not Alpha.__is_good_card_for_expanding(steps[_idx].card):
            if np.sum(scores[_idx, :3]) < steps[_idx].card.size * 3:
                return False
        if np.sum(scores[_idx, :3]) < steps[_idx].card.size * 2:
            return False
        return True

    @staticmethod
    def __score_consolidating_step(status: Status, step: Step, remaining_sp_card: int):
        next_stage = util.estimate_stage(status.stage, step)
        my_sp = status.my_sp + util.estimate_my_sp_diff(status.stage, next_stage, step)
        his_sp_diff = util.estimate_his_sp_diff(status.stage, next_stage, step)
        estimated_occupied_grids_1 = Evaluation.occupied_grids(next_stage, my_dilate=1, his_dilate=0, connectivity=8)
        area = len(next_stage.my_ink)
        dilated_area = Evaluation.dilated_area(next_stage)

        sp_card_penalty = 0
        if not util.is_special_card(step.card):
            pattern = step.card.get_pattern(step.rotate)
            pos = pattern.offset[pattern.squares == Grid.MySpecial.value] + step.pos
            distance = Evaluation.square_distance(status.stage, pos)
        else:
            distance = 0
            if remaining_sp_card <= 1:
                sp_card_penalty = -18
            elif remaining_sp_card <= 2:
                sp_card_penalty = -12

        return my_sp * 6, area, estimated_occupied_grids_1 * 0.6, distance * -0.01, his_sp_diff * -4, sp_card_penalty, dilated_area * -0.1

    @staticmethod
    def __score_special_attack_step(status: Status, step: Step, sp_threshold):
        next_stage = util.estimate_stage(status.stage, step)
        my_ink = len(next_stage.my_ink)
        his_ink = len(next_stage.his_ink)
        sp = status.my_sp + util.estimate_my_sp_diff(status.stage, next_stage, step)
        if sp > sp_threshold:
            sp = -sp
        drop_card = Evaluation.drop_card_penalty(status, step)
        return my_ink, his_ink * -1, sp * 6, drop_card

    @staticmethod
    def __score_round_2_step(status: Status, step: Step):
        next_stage = util.estimate_stage(status.stage, step)
        my_ink = len(next_stage.my_ink)
        his_ink = len(next_stage.his_ink)
        sp = status.my_sp + util.estimate_my_sp_diff(status.stage, next_stage, step)
        return my_ink, his_ink * -1, sp

    @staticmethod
    def __score_round_1_step(status: Status, step: Step):
        next_stage = util.estimate_stage(status.stage, step)
        my_ink = len(next_stage.my_ink)
        his_ink = len(next_stage.his_ink)
        return my_ink, his_ink * -1

    @staticmethod
    def __sort_hands_for_expanding(status: Status) -> List[Card]:
        good_cards = [card for card in status.hands if len(status.get_possible_steps(card, action=Step.Action.Place)) > 0 and card.size > 3]
        sorted_cards = sorted(good_cards, key=lambda c: (c.size, max(c.get_pattern().width, c.get_pattern().height)), reverse=True)
        sp_cards = [card for card in sorted_cards if util.is_special_card(card)]
        not_sp_cards = [card for card in sorted_cards if not util.is_special_card(card)]
        good_not_sp_cards = [card for card in not_sp_cards if Alpha.__is_good_card_for_expanding(card)]
        bad_not_sp_cards = [card for card in not_sp_cards if not Alpha.__is_good_card_for_expanding(card)]
        return good_not_sp_cards + sp_cards + bad_not_sp_cards

    @staticmethod
    def __is_good_card_for_expanding(card: Card) -> bool:
        return card.size >= 9 or card.get_pattern().height >= 6 or card.get_pattern().width >= 6

    @staticmethod
    def __sort_hands_for_consolidating(status: Status) -> List[Card]:
        good_cards = [card for card in status.hands if len(status.get_possible_steps(card, action=Step.Action.Place)) > 0]
        sorted_cards = sorted(good_cards, key=lambda c: c.size, reverse=True)
        sp_cards = [card for card in sorted_cards if util.is_special_card(card)]
        not_sp_cards = [card for card in sorted_cards if not util.is_special_card(card)]
        return not_sp_cards + sp_cards

    def reset(self):
        return


class Evaluation:
    @staticmethod
    def occupied_grids(stage: Stage, my_dilate=0, his_dilate=0, connectivity=4):
        """
        Return the number of occupied squares that is not connected with opponent's squares.
        """
        grid = stage.grid.copy()
        if his_dilate > 0:
            his_mask = (np.bitwise_and(grid, Grid.HisInk.value | Grid.HisSpecial.value) > 0).astype(np.uint8) * 255
            his_mask = cv2.dilate(his_mask, kernel=np.ones((his_dilate * 2 + 1, his_dilate * 2 + 1), dtype=np.uint8))
            his_mask = np.bitwise_and(his_mask == 255, grid == Grid.Empty.value)
            grid[his_mask] = Grid.HisInk.value
        if my_dilate > 0:
            my_mask = (np.bitwise_and(grid, Grid.MyInk.value | Grid.MySpecial.value) > 0).astype(np.uint8) * 255
            my_mask = cv2.dilate(my_mask, kernel=np.ones((my_dilate * 2 + 1, my_dilate * 2 + 1), dtype=np.uint8))
            my_mask = np.bitwise_and(my_mask == 255, grid == Grid.Empty.value)
            grid[my_mask] = Grid.MyInk.value
        his_ink_idx = np.argwhere(np.bitwise_and(grid, Grid.HisInk.value | Grid.HisSpecial.value))
        empty_idx = np.argwhere(grid == Grid.Empty.value)
        mask = np.zeros(stage.shape, dtype=np.uint8)
        mask[his_ink_idx[:, 0], his_ink_idx[:, 1]] = 255
        mask[empty_idx[:, 0], empty_idx[:, 1]] = 255
        # opencv bug: https://github.com/opencv/opencv-python/issues/602
        # num_labels, labels = cv2.connectedComponents(mask, connectivity=connectivity)
        try:
            num_labels, labels = cv2.connectedComponents(mask, connectivity=connectivity)
        except Exception:
            logger.error(f'tableturf.ai.alpha.next_step: failed to calculate connectedComponents with connectivity={connectivity}')
            num_labels, labels = cv2.connectedComponents(mask)
        unoccupied_labels = np.concatenate([[0], np.unique(labels[his_ink_idx[:, 0], his_ink_idx[:, 1]])])  # background + connected with his inks
        return np.sum(np.isin(labels, unoccupied_labels, invert=True)) + np.sum(np.bitwise_and(grid, Grid.MyInk.value | Grid.MySpecial.value) > 0)

    @staticmethod
    def conflict_grids(stage: Stage, my_dilate, his_dilate):
        """
        Return the number of conflict squares if dilate our squares.
        """
        grid = stage.grid.copy()
        if his_dilate > 0:
            mask = (np.bitwise_and(grid, Grid.HisInk.value | Grid.HisSpecial.value) > 0).astype(np.uint8) * 255
            mask = cv2.dilate(mask, kernel=np.ones((his_dilate * 2 + 1, his_dilate * 2 + 1), dtype=np.uint8))
            his_overlap = np.bitwise_and(mask == 255, np.bitwise_and(grid, Grid.MyInk.value | Grid.MySpecial.value) > 0)
            mask = np.bitwise_and(mask == 255, grid == Grid.Empty.value)
            grid[mask] = Grid.HisInk.value
        mask = (np.bitwise_and(grid, Grid.MyInk.value | Grid.MySpecial.value) > 0).astype(np.uint8) * 255
        mask = cv2.dilate(mask, kernel=np.ones((my_dilate * 2 + 1, my_dilate * 2 + 1), dtype=np.uint8))
        overlap = np.bitwise_and(mask == 255, np.bitwise_and(grid, Grid.HisInk.value | Grid.HisSpecial.value) > 0)
        if his_dilate > 0:
            overlap = np.bitwise_or(his_overlap, overlap)
        return np.sum(overlap)

    @staticmethod
    def dilated_area(stage: Stage, dilate=1):
        mask = (np.bitwise_and(stage.grid, Grid.MyInk.value | Grid.MySpecial.value) > 0).astype(np.uint8) * 255
        mask = cv2.dilate(mask, kernel=np.ones((dilate * 2 + 1, dilate * 2 + 1), dtype=np.uint8))
        mask = np.bitwise_and(mask == 255, stage.grid == Grid.Empty.value)
        return np.sum(mask) + len(stage.my_ink)

    @staticmethod
    def ink_size(stage: Stage) -> float:
        return np.linalg.norm(np.max(stage.my_ink, axis=0) - np.min(stage.my_ink, axis=0))

    @staticmethod
    def square_distance(stage: Stage, pos: np.ndarray) -> float:
        return np.min(np.linalg.norm(stage.his_ink - pos[np.newaxis, ...], axis=1))

    @staticmethod
    def possible_steps(next_status: Status) -> float:
        cards = next_status.hands
        return np.sum([len(next_status.get_possible_steps(card=card, action=Step.Action.Place)) * np.power(2, card.size) for card in cards])

    @staticmethod
    def drop_card_penalty(status: Status, step: Step):
        if step.action != Step.Action.Skip:
            return 0
        if step.card.sp_cost - status.my_sp <= 1:
            return -step.card.size
        return -step.card.size / step.card.sp_cost + 0.1 * step.card.sp_cost

    @staticmethod
    def recursive_area(status: Status, step: Step, depth: int):
        if depth == 1:
            return len(util.estimate_stage(status.stage, step).my_ink)
        next_status = util.estimate_status(status, step, expand=True)
        score = np.mean([np.max([Evaluation.recursive_area(_status, _step, depth - 1) for _step in _status.get_possible_steps()]) for _status in next_status])
        return score
