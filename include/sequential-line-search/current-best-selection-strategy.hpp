#ifndef SEQUENTIAL_LINE_SEARCH_CURRENT_BEST_SELECTION_STRATEGY_HPP
#define SEQUENTIAL_LINE_SEARCH_CURRENT_BEST_SELECTION_STRATEGY_HPP

namespace sequential_line_search
{
    /// \brief Strategy for selecting the so-far-observed current best point.
    enum class CurrentBestSelectionStrategy
    {
        LargestExpectValue, /// Select the point that has the largest expected value, x^{+}, as in [Koyama+17].
        LastSelection,      /// Select the point that was chosen in the last subtask, x^{chosen}, as suggested in
                            /// [Koyama+20].
    };
} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_CURRENT_BEST_SELECTION_STRATEGY_HPP
