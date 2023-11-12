package my.aplication.manejointeligentedepragas.RecyclerViewAdapter;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Color;
//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.v7.widget.CardView;
import androidx.cardview.widget.CardView;
//import android.support.v7.widget.RecyclerView;
import androidx.recyclerview.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import my.aplication.manejointeligentedepragas.AplicaMetodoDeControle;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PresencaPraga;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;
import my.aplication.manejointeligentedepragas.PlanoDeAmostragem;
import my.aplication.manejointeligentedepragas.Pragas;

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.model.PragaModel;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

public class PragaCardAdapter extends RecyclerView.Adapter<PragaCardAdapter.ViewHolder>{

    private static final String TAG = "PragaCardAdapter";

    private ArrayList<PragaModel> cards = new ArrayList<>();
    private Context pragaContext;
    private int codCultura;
    private int Cod_Talhao;
    private String NomeTalhao;
    private String nome;
    private int Cod_Propriedade;
    private boolean aplicado;
    private String nomePropriedade;
    private int Cod_Planta;

    public PragaCardAdapter(Context pragaContext, ArrayList<PragaModel> cards,int Cod_Talhao, String NomeTalhao, int codCultura, String nome, int Cod_Propriedade, boolean aplicado,String nomePropriedade, int Cod_Planta) {
        this.cards = cards;
        this.pragaContext = pragaContext;
        this.Cod_Talhao = Cod_Talhao;
        this.NomeTalhao = NomeTalhao;
        this.codCultura = codCultura;
        this.nome = nome;
        this.Cod_Propriedade = Cod_Propriedade;
        this.aplicado = aplicado;
        this.nomePropriedade = nomePropriedade;
        this.Cod_Planta = Cod_Planta;
    }

    @NonNull
    @Override
    // resposável por inflar a view

    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.card_praga, parent, false);
        ViewHolder holder = new ViewHolder(view);
        return holder;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, final int position) {
        Log.d(TAG, "onBindViewHolder: called.");


        holder.praga.setText(cards.get(position).getNome());
        Controller_Usuario cu = new Controller_Usuario(pragaContext);
        holder.tipoUsu = cu.getUser().getTipo();

        if(cards.get(position).getStatus() == 0){
            holder.cardView.setCardBackgroundColor(Color.parseColor("#659251"));
        }else if(cards.get(position).getStatus() == 1){
            holder.cardView.setCardBackgroundColor(Color.parseColor("#ECC911"));
            //#FFDAC623
        }else if(cards.get(position).getStatus() == 2){
            holder.cardView.setCardBackgroundColor(Color.parseColor("#FD991111"));
        }

        // true para, false continua pra proxima tela. função para click demorado
        if(holder.tipoUsu.equals("Produtor")) {
            holder.parent_layout_praga.setOnLongClickListener(new View.OnLongClickListener() {
                @Override
                public boolean onLongClick(View v) {
                    exibirCaixaDialogo(cards.get(position));
                    return true;
                }
            });
        }
        holder.parent_layout_praga.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {


                if(cards.get(position).getStatus() == 0){
                    exibirCaixaDialogoVerde(cards.get(position));
                }else if(cards.get(position).getStatus() == 1){
                    if(aplicado){
                        exibirCaixaDialogoAmarela(cards.get(position));
                    }else{
                        Intent i = new Intent(pragaContext, PlanoDeAmostragem.class);
                        i.putExtra("Cod_Talhao", Cod_Talhao);
                        i.putExtra("NomeTalhao", NomeTalhao);
                        i.putExtra("Cod_Praga", cards.get(position).getCod_Praga());
                        i.putExtra("nomePraga", cards.get(position).getNome());
                        i.putExtra("Cod_Cultura", codCultura);
                        i.putExtra("NomeCultura", nome);
                        i.putExtra("Cod_Propriedade", Cod_Propriedade);
                        i.putExtra("Aplicado", aplicado);
                        i.putExtra("nomePropriedade", nomePropriedade);
                        i.putExtra("Cod_Planta", Cod_Planta);
                        pragaContext.startActivity(i);
                    }
                }else if(cards.get(position).getStatus() == 2){
                    if(aplicado){
                        exibirCaixaDialogoVermelha(cards.get(position));
                    }else{
                        Intent i = new Intent(pragaContext, AplicaMetodoDeControle.class);
                        i.putExtra("Cod_Talhao", Cod_Talhao);
                        i.putExtra("NomeTalhao", NomeTalhao);
                        i.putExtra("Cod_Praga", cards.get(position).getCod_Praga());
                        i.putExtra("Cod_Cultura", codCultura);
                        i.putExtra("NomeCultura", nome);
                        i.putExtra("Cod_Propriedade", Cod_Propriedade);
                        i.putExtra("Aplicado", aplicado);
                        i.putExtra("nomePropriedade", nomePropriedade);
                        i.putExtra("Cod_Planta", Cod_Planta);
                        pragaContext.startActivity(i);
                    }
                }
                // chama a intent nesse adapter
                // pega o context do construtor
                //int codigo = cards.get(position).getCod_Propriedade();
                //i.putExtra("Cod_Propriedade", codigo);
                //pragaContext.startActivity(i);

            }
        });

    }

    @Override
    public int getItemCount() {
        return cards.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {

        TextView praga;
        CardView cardView;

        String tipoUsu;

        RelativeLayout parent_layout_praga;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);

            praga = itemView.findViewById(R.id.tvNomePraga);
            cardView = itemView.findViewById(R.id.CardPragas);
            parent_layout_praga = itemView.findViewById(R.id.parent_layout_praga);

        }
    }

    public void exibirCaixaDialogo(final PragaModel p)
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(pragaContext);
        dlgBox.setTitle("Excluindo...");
        dlgBox.setMessage("Tem certeza que deseja excluir essa praga?");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {

                Utils u = new Utils();
                if(!u.isConected(pragaContext)) {
                    Toast.makeText(pragaContext, "Você só pode realizar essa ação enquanto estiver online!",Toast.LENGTH_LONG).show();
                }else{
                    ExcluirPraga(p.getCod_Praga(), Cod_Talhao);
                }
            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nada
            }
        });

        dlgBox.show();

    }

    public void exibirCaixaDialogoVerde(final PragaModel p)
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(pragaContext);
        dlgBox.setTitle("Aviso!");
        dlgBox.setMessage("Essa praga está controlada, deseja fazer uma contagem para fins de monitoramento? (caso seja constatada a necessidade, uma nova aplicação deverá ser realizada)");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Intent i = new Intent(pragaContext, PlanoDeAmostragem.class);
                i.putExtra("Cod_Praga", p.getCod_Praga());
                i.putExtra("nomePraga", p.getNome());
                i.putExtra("Cod_Cultura", codCultura);
                i.putExtra("NomeCultura", nome);
                i.putExtra("Cod_Propriedade", Cod_Propriedade);
                i.putExtra("Aplicado", aplicado);
                i.putExtra("nomePropriedade", nomePropriedade);
                i.putExtra("Cod_Talhao", Cod_Talhao);
                i.putExtra("NomeTalhao", NomeTalhao);
                i.putExtra("Cod_Planta", Cod_Planta);
                pragaContext.startActivity(i);
            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nada
            }
        });

        dlgBox.show();

    }

    public void exibirCaixaDialogoAmarela(final PragaModel p)
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(pragaContext);
        dlgBox.setTitle("Aviso!");
        dlgBox.setMessage("Aplicação realizada recentemente, deseja fazer uma contagem para fins de monitoramento? (caso seja constatada a necessidade, uma nova aplicação deverá ser realizada)");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Intent i = new Intent(pragaContext, PlanoDeAmostragem.class);
                i.putExtra("Cod_Talhao", Cod_Talhao);
                i.putExtra("NomeTalhao", NomeTalhao);
                i.putExtra("Cod_Praga", p.getCod_Praga());
                i.putExtra("nomePraga", p.getNome());
                i.putExtra("Cod_Cultura", codCultura);
                i.putExtra("NomeCultura", nome);
                i.putExtra("Cod_Propriedade", Cod_Propriedade);
                i.putExtra("Aplicado", aplicado);
                i.putExtra("nomePropriedade", nomePropriedade);
                i.putExtra("Cod_Planta", Cod_Planta);
                pragaContext.startActivity(i);
            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nada
            }
        });

        dlgBox.show();

    }

    public void exibirCaixaDialogoVermelha(final PragaModel p)
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(pragaContext);
        dlgBox.setTitle("Aviso!");
        dlgBox.setMessage("Aplicação realizada recentemente. Aguarde o intervalo recomendado entre as aplicações para evitar fitotoxidez na cultura.");
        dlgBox.setPositiveButton("Ok", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nada
            }
        });


        dlgBox.show();

    }

    public void ExcluirPraga(final Integer CodP, final Integer CodT){

        String url = "https://mip.software/phpapp/excluirPraga.php?Cod_Praga="+ CodP +"&&Cod_Talhao="+CodT ;

        RequestQueue queue = Volley.newRequestQueue(pragaContext);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {
                //Parsing json
                //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                try {
                    JSONObject obj = new JSONObject(response);
                    boolean confirmacao = obj.getBoolean("confirmacao");
                    if(confirmacao){

                        Controller_PresencaPraga cpp = new Controller_PresencaPraga(pragaContext);
                        cpp.removerPresencaPragaEspecifica(CodT, CodP);

                        Intent i = new Intent(pragaContext, Pragas.class);
                        Toast.makeText(pragaContext, "Praga excluída com sucesso!",Toast.LENGTH_LONG).show();
                        i.putExtra("Cod_Talhao", Cod_Talhao);
                        i.putExtra("NomeTalhao", NomeTalhao);
                        i.putExtra("Cod_Talhao", Cod_Talhao);
                        i.putExtra("NomeTalhao", NomeTalhao);
                        i.putExtra("Cod_Cultura", codCultura);
                        i.putExtra("NomeCultura", nome);
                        i.putExtra("Cod_Propriedade", Cod_Propriedade);
                        i.putExtra("nomePropriedade", nomePropriedade);
                        i.putExtra("Cod_Planta", Cod_Planta);
                        pragaContext.startActivity(i);
                    }else{
                        Toast.makeText(pragaContext, "Praga não excluída! Tente novamente",Toast.LENGTH_LONG).show();
                    }

                } catch (JSONException e) {
                    Toast.makeText(pragaContext, e.toString(), Toast.LENGTH_LONG).show();
                }
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(pragaContext,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }
}
