package my.aplication.manejointeligentedepragas.VisualizaAdapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.model.PragaModel;

import java.util.ArrayList;

public class VisuPragasAdapter extends ArrayAdapter<PragaModel> {

    private ArrayList<PragaModel> cards = new ArrayList<>();
    private Context pragaContext;

    public VisuPragasAdapter( Context pragaContext,ArrayList<PragaModel> cards) {
        super(pragaContext, R.layout.construtor_visu_pragas,cards);
        this.cards = cards;
        this.pragaContext = pragaContext;
    }


    @Override
    public View getView(int position,  View convertView, ViewGroup parent) {


        LayoutInflater inflater = (LayoutInflater) pragaContext
                .getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View rowVIew = inflater.inflate(R.layout.construtor_visu_pragas,parent, false);


        TextView nomePraga;
        TextView nomeCientificoPraga;

        nomePraga = rowVIew.findViewById(R.id.tvNomePraga);
        nomeCientificoPraga = rowVIew.findViewById(R.id.tvNomeCientPraga);

        nomePraga.setText(cards.get(position).getNome());
        nomeCientificoPraga.setText(cards.get(position).getNomeCientifico());

        return rowVIew;
    }


}
